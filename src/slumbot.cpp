#include "texas/slumbot.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include "texas/features.hpp"
#include "texas/rollout.hpp"

namespace texas {

namespace {

size_t writeCb(char* ptr, size_t size, size_t nmemb, void* userdata) {
    static_cast<std::string*>(userdata)->append(ptr, size * nmemb);
    return size * nmemb;
}

bool postJsonOnce(const std::string& host, const std::string& path, const std::string& body,
                  std::string& out, long& status, std::string& why) {
    out.clear();
    CURL* h = curl_easy_init();
    if (h == nullptr) {
        why = "curl init";
        return false;
    }
    const std::string url = "https://" + host + path;
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    struct curl_slist* hdrs = nullptr;
    hdrs = curl_slist_append(hdrs, "Content-Type: application/json");
    hdrs = curl_slist_append(hdrs, "Accept: application/json");
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, hdrs);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, writeCb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 10L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    const CURLcode rc = curl_easy_perform(h);
    status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(hdrs);
    curl_easy_cleanup(h);
    if (rc != CURLE_OK) {
        why = std::string("curl=") + curl_easy_strerror(rc);
        return false;
    }
    if (status != 200) {
        why = "http=" + std::to_string(status);
        // 把服务器的错误正文带出来，便于分辨限流还是逻辑错误。
        const std::string snippet = out.substr(0, std::min<size_t>(out.size(), 200));
        if (!snippet.empty()) why += " body=" + snippet;
        return false;
    }
    return true;
}

// 网络抖动/限流（429、5xx、连接错误）值得重试；4xx 逻辑错误不重试。
bool postJson(const std::string& host, const std::string& path, const std::string& body,
              std::string& out, int attempts, std::string& why) {
    why.clear();
    for (int a = 0; a < attempts; ++a) {
        long status = 0;
        std::string w;
        if (postJsonOnce(host, path, body, out, status, w)) return true;
        why = w;
        const bool retryable = (status == 0 || status == 429 || status >= 500);
        if (!retryable || a + 1 >= attempts) return false;
        // 指数退避：200ms、800ms……给服务器留恢复时间。
        std::this_thread::sleep_for(std::chrono::milliseconds(200 * (1 << a)));
    }
    return false;
}

// ---- 极简 JSON 取值：Slumbot 的字段固定、没有嵌套对象，够用即可。 ----

bool fieldPos(const std::string& j, const char* key, size_t& pos) {
    const std::string pat = std::string("\"") + key + "\"";
    size_t p = j.find(pat);
    if (p == std::string::npos) return false;
    p = j.find(':', p + pat.size());
    if (p == std::string::npos) return false;
    pos = p + 1;
    return true;
}

bool jsonString(const std::string& j, const char* key, std::string& out) {
    size_t p = 0;
    if (!fieldPos(j, key, p)) return false;
    p = j.find('"', p);
    if (p == std::string::npos) return false;
    const size_t e = j.find('"', p + 1);
    if (e == std::string::npos) return false;
    out = j.substr(p + 1, e - p - 1);
    return true;
}

bool jsonInt(const std::string& j, const char* key, long long& out) {
    size_t p = 0;
    if (!fieldPos(j, key, p)) return false;
    while (p < j.size() && !(j[p] == '-' || j[p] == '+' || (j[p] >= '0' && j[p] <= '9'))) ++p;
    if (p >= j.size()) return false;
    char* end = nullptr;
    out = std::strtoll(j.c_str() + p, &end, 10);
    return end != j.c_str() + p;
}

bool jsonStringArray(const std::string& j, const char* key, std::vector<std::string>& out) {
    size_t p = 0;
    if (!fieldPos(j, key, p)) return false;
    p = j.find('[', p);
    if (p == std::string::npos) return false;
    ++p;
    while (p < j.size() && (j[p] == ' ' || j[p] == '\t' || j[p] == '\n' || j[p] == '\r')) ++p;
    if (p < j.size() && j[p] == ']') return true;  // 空数组
    while (p < j.size()) {
        const size_t q = j.find('"', p);
        if (q == std::string::npos) break;
        const size_t e = j.find('"', q + 1);
        if (e == std::string::npos) break;
        out.push_back(j.substr(q + 1, e - q - 1));
        p = e + 1;
        const size_t close = j.find(']', p);
        const size_t comma = j.find(',', p);
        if (close != std::string::npos && (comma == std::string::npos || close < comma)) break;
    }
    return true;
}

// 河牌前全下的"EV 调整"：用全下那一刻的胜率折算底池，剔掉后面的发牌运气。
// 已知双方底牌 + 已发公共牌，剩余公共牌全枚举（最多 C(48,2)=1128 种），
// 精确算胜率。返回 true 表示这手确实被调整了。
bool allinAdjustedImpl(const std::string& action, const Card mine[2], const Card bot[2],
                       const Card board[5], int clientPos, long long winnings, double& adjusted) {
    // 已知双方底牌，直接用 makeState 重建；boardN 从 0 开始，由 applyMove
    // 在街道切换时推进（公共牌本身已经写进 state，不需要重新发）。
    HUState st = (clientPos == 0) ? makeState(bot, mine, board, 0)
                                  : makeState(mine, bot, board, 0);
    Rng dummy(1);
    size_t i = 0;
    while (i < action.size()) {
        const char c = action[i++];
        if (c == '/') continue;
        Move m{};
        if (c == 'k') {
            m.kind = MK_CHECK;
            m.to = st.contrib[st.toAct];
        } else if (c == 'c') {
            m.kind = MK_CALL;
            m.to = st.contrib[1 - st.toAct];
        } else if (c == 'f') {
            m.kind = MK_FOLD;
            m.to = st.contrib[st.toAct];
        } else if (c == 'b') {
            int v = 0;
            const size_t start = i;
            while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                v = v * 10 + (action[i] - '0');
                ++i;
            }
            if (i == start) return false;
            m.kind = MK_RAISE;
            m.to = static_cast<int16_t>(v);
        } else {
            return false;
        }
        applyMove(st, m, dummy);
        if (st.finished) break;
    }
    // 只有河牌前双方全下才需要调整；河牌全下的结果已经由最终牌面决定。
    if (!(st.allin[0] && st.allin[1]) || st.street >= ST_RIVER) return false;
    const int mySeat = (clientPos == 0) ? 1 : 0;
    const int boardN = (st.street == ST_PREFLOP) ? 0 : int(st.street) + 2;
    const int pot = st.pot();
    const int invested = int(st.total[mySeat]);

    bool used[kNumCards] = {};
    used[mine[0]] = used[mine[1]] = used[bot[0]] = used[bot[1]] = true;
    Card known[5];
    for (int k = 0; k < boardN; ++k) {
        known[k] = board[k];
        used[board[k]] = true;
    }
    Card deck[kNumCards];
    int nDeck = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[c]) deck[nDeck++] = static_cast<Card>(c);
    }

    long long win = 0, tie = 0, total = 0;
    // 枚举剩余公共牌（1 或 2 张），不使用实际发牌结果。
    const int need = 5 - boardN;
    auto evalOne = [&](const Card* runout, int nRun) {
        Card a[7], b[7];
        a[0] = mine[0];
        a[1] = mine[1];
        b[0] = bot[0];
        b[1] = bot[1];
        int n = 2;
        for (int k = 0; k < boardN; ++k) {
            a[n] = b[n] = known[k];
            ++n;
        }
        for (int k = 0; k < nRun; ++k) {
            a[n] = b[n] = runout[k];
            ++n;
        }
        const int s0 = evaluate7(a);
        const int s1 = evaluate7(b);
        if (s0 > s1) ++win;
        else if (s0 == s1) ++tie;
        ++total;
    };
    if (need <= 0) {
        evalOne(nullptr, 0);
    } else if (need == 1) {
        for (int x = 0; x < nDeck; ++x) evalOne(&deck[x], 1);
    } else if (need == 2) {
        for (int x = 0; x < nDeck; ++x) {
            for (int y = x + 1; y < nDeck; ++y) {
                const Card run[2] = {deck[x], deck[y]};
                evalOne(run, 2);
            }
        }
    } else {
        // 翻前全下要发 5 张：全枚举 C(48,5)≈171 万太贵。抽 2048 组跑完，
        // 胜率误差约 1%（换到底池上 ~±400 筹码），远小于发牌本身的方差。
        Rng sampler(0x51A1B07ULL + uint64_t(need));
        const int samples = 2048;
        Card run[5];
        for (int s = 0; s < samples; ++s) {
            // 部分 Fisher-Yates：不重复地抽 need 张（每次从原牌堆重新开始，
            // 保证各样本独立且边缘分布均匀）。
            Card tmp[kNumCards];
            std::memcpy(tmp, deck, sizeof(Card) * static_cast<size_t>(nDeck));
            for (int k = 0; k < need; ++k) {
                const int j = k + static_cast<int>(sampler() % static_cast<size_t>(nDeck - k));
                std::swap(tmp[k], tmp[j]);
                run[k] = tmp[k];
            }
            evalOne(run, need);
        }
    }
    if (total <= 0) return false;
    const double equity = (double(win) + 0.5 * double(tie)) / double(total);
    adjusted = equity * double(pot) - double(invested);
    (void)winnings;
    return true;
}

// 全下纪律 guard：外部明细显示我们的全下跟注平均调整后 EV 约 -1BB/手
// （每手 -99BB），而 Slumbot 的全下几乎总是顶范围。手牌百分位不够就弃牌，
// 只保留顶范围跟注。跟注额太小的全下不拦（赔率通常已经足够）。
bool allinGuardFoldImpl(const HUState& st, float prePct, int postCat, float minRisk) {
    if (st.finished) return false;
    const int me = st.toAct;
    const int need = toCall(st, me);
    if (need <= 0) return false;
    const bool facingAllin = st.allin[1 - me] != 0 || need >= st.stack[me];
    if (!facingAllin) return false;
    if (double(need) < double(minRisk) * double(st.stack[me])) return false;
    const bool pre = (st.street == ST_PREFLOP);
    if (pre) {
        if (prePct <= 0.0f) return false;
        int hole[2] = {int(st.hole[me][0]), int(st.hole[me][1])};
        int bd[5] = {0, 0, 0, 0, 0};
        return handStrength(hole, bd, 0) < prePct;
    }
    if (postCat < 0) return false;
    // 翻后用"最佳五张的牌力类别"而不是百分位：百分位会把两对算成 90%+，
    // 而外部数据显示两对及以下跟全下几乎都在送钱。
    const int boardN = int(st.boardN);
    int cards[7];
    int nc = 0;
    cards[nc++] = int(st.hole[me][0]);
    cards[nc++] = int(st.hole[me][1]);
    for (int i = 0; i < boardN; ++i) cards[nc++] = int(st.board[i]);
    return handCategoryOf(evaluateBest(cards, nc)) < postCat;
}

std::string moveToIncr(const Move& m) {
    switch (m.kind) {
        case MK_FOLD: return "f";
        case MK_CHECK: return "k";
        case MK_CALL: return "c";
        default: return "b" + std::to_string(int(m.to));
    }
}

}  // namespace

bool allinAdjustedWinnings(const std::string& action, const Card mine[2], const Card bot[2],
                           const Card board[5], int clientPos, long long winnings,
                           double& adjusted) {
    return allinAdjustedImpl(action, mine, bot, board, clientPos, winnings, adjusted);
}

bool allinGuardFold(const HUState& st, float prePct, int postCat, float minRisk) {
    return allinGuardFoldImpl(st, prePct, postCat, minRisk);
}

namespace {

// 重放动作串。lastOpp 非空时，把"对手（clientPos 座位）最近一次决策"的
// 动作前状态与实际动作槽记下来 —— rangeEq 的似然要在这个状态上算。
bool buildStateImpl(const std::string& action, const std::string& hole0,
                    const std::string& hole1, const std::vector<std::string>& board,
                    int clientPos, Rng& rng, HUState& out, RangeObs* lastOpp) {
    if (hole0.empty() || hole1.empty()) return false;
    const Card mine[2] = {parseCard(hole0.c_str()), parseCard(hole1.c_str())};
    if (mine[0] < 0 || mine[1] < 0 || mine[0] == mine[1]) return false;

    bool used[kNumCards] = {};
    used[mine[0]] = used[mine[1]] = true;

    Card boardCards[5] = {0, 0, 0, 0, 0};
    bool filled[5] = {false, false, false, false, false};
    const int boardN = std::min<int>(5, static_cast<int>(board.size()));
    for (int i = 0; i < boardN; ++i) {
        const Card c = parseCard(board[static_cast<size_t>(i)].c_str());
        if (c < 0 || used[c]) return false;
        boardCards[i] = c;
        filled[i] = true;
        used[c] = true;
    }

    // 对手底牌和未发公共牌都用任意未用牌填：策略只看得见自己的牌 + 已发公共牌，
    // perfectEquity 也只用 boardN 张，所以这些占位不影响选动作。
    Card rest[kNumCards];
    int nr = 0;
    for (int c = 0; c < kNumCards; ++c) {
        if (!used[c]) rest[nr++] = static_cast<Card>(c);
    }
    if (nr < 2) return false;
    int ri = 0;
    const Card opp[2] = {rest[ri++], rest[ri++]};
    for (int i = 0; i < 5; ++i) {
        if (!filled[i]) boardCards[i] = rest[ri++];
    }

    HUState st = (clientPos == 0) ? makeState(opp, mine, boardCards, boardN)
                                  : makeState(mine, opp, boardCards, boardN);
    if (lastOpp != nullptr) *lastOpp = RangeObs{};
    size_t i = 0;
    while (i < action.size()) {
        const char c = action[i++];
        if (c == '/') continue;
        Move m{};
        if (c == 'k') {
            m.kind = MK_CHECK;
            m.to = st.contrib[st.toAct];
        } else if (c == 'c') {
            m.kind = MK_CALL;
            m.to = st.contrib[1 - st.toAct];
        } else if (c == 'f') {
            m.kind = MK_FOLD;
            m.to = st.contrib[st.toAct];
        } else if (c == 'b') {
            int v = 0;
            const size_t start = i;
            while (i < action.size() && action[i] >= '0' && action[i] <= '9') {
                v = v * 10 + (action[i] - '0');
                ++i;
            }
            if (i == start) return false;
            m.kind = MK_RAISE;
            m.to = static_cast<int16_t>(v);
        } else {
            return false;
        }
        // 座位约定：clientPos = 0 时我们是 BB（seat 1），对手是 seat 0。
        if (lastOpp != nullptr && st.toAct == clientPos) {
            lastOpp->pre = st;
            lastOpp->slot = slotOfMove(st, m);
            lastOpp->valid = lastOpp->slot >= 0;
        }
        applyMove(st, m, rng);
        if (st.finished) break;
    }
    out = st;
    return true;
}

}  // namespace

bool buildSlumbotState(const std::string& action, const std::string& hole0,
                       const std::string& hole1, const std::vector<std::string>& board,
                       int clientPos, Rng& rng, HUState& out) {
    return buildStateImpl(action, hole0, hole1, board, clientPos, rng, out, nullptr);
}

bool slumbotLastOppDecision(const std::string& action, const std::string& hole0,
                            const std::string& hole1, const std::vector<std::string>& board,
                            int clientPos, Rng& rng, RangeObs& out) {
    HUState st;
    RangeObs last;
    if (!buildStateImpl(action, hole0, hole1, board, clientPos, rng, st, &last)) return false;
    out = last;
    return last.valid;
}

SlumbotResult playSlumbot(const nn::Actor actors[2], const SlumbotConfig& cfg) {
    SlumbotResult res;
    Rng rng(cfg.seed);
    std::string token;
    int consecutiveErrors = 0;
    double sumW = 0.0;
    double sumSq = 0.0;
    double sumAdj = 0.0;
    double sumAdjSq = 0.0;
    FILE* log = nullptr;
    if (!cfg.logPath.empty()) {
        log = std::fopen(cfg.logPath.c_str(), "w");
        if (log != nullptr) {
            std::fprintf(log,
                         "hand,client_pos,winnings,adjusted,pot,allin_adjusted,action,"
                         "hole,bot_hole,board\n");
        }
    }

    for (int h = 1; h <= cfg.hands; ++h) {
        ++res.attempted;
        const std::string body = token.empty() ? "{}" : ("{\"token\": \"" + token + "\"}");
        std::string resp;
        std::string why;
        if (!postJson(cfg.host, "/slumbot/api/new_hand", body, resp, 3, why)) {
            ++res.errors;
            token.clear();
            std::printf("new_hand 失败: %s\n", why.c_str());
            std::fflush(stdout);
            if (++consecutiveErrors >= 3) {
                std::printf("new_hand 连续失败，停止\n");
                break;
            }
            continue;
        }
        consecutiveErrors = 0;

        bool finished = false;
        int steps = 0;
        while (!finished && steps++ < 64) {
            std::string err;
            if (jsonString(resp, "error_msg", err)) {
                ++res.errors;
                std::printf("slumbot error: %s\n", err.c_str());
                token.clear();
                break;
            }
            std::string nt;
            if (jsonString(resp, "token", nt)) token = nt;

            std::string action;
            jsonString(resp, "action", action);
            std::vector<std::string> hole;
            std::vector<std::string> board;
            std::vector<std::string> botHole;
            jsonStringArray(resp, "hole_cards", hole);
            jsonStringArray(resp, "board", board);
            jsonStringArray(resp, "bot_hole_cards", botHole);
            long long clientPos = 0;
            jsonInt(resp, "client_pos", clientPos);

            long long winnings = 0;
            if (jsonInt(resp, "winnings", winnings)) {
                ++res.hands;
                res.winnings += winnings;
                sumW += double(winnings);
                sumSq += double(winnings) * double(winnings);
                // 全下 EV 调整：用全下时的胜率折算底池，剔掉发牌运气。
                double adj = double(winnings);
                bool didAdjust = false;
                if (hole.size() >= 2 && botHole.size() >= 2 && board.size() <= 5) {
                    Card mine[2] = {parseCard(hole[0].c_str()), parseCard(hole[1].c_str())};
                    Card theirs[2] = {parseCard(botHole[0].c_str()),
                                      parseCard(botHole[1].c_str())};
                    Card bd[5] = {0, 0, 0, 0, 0};
                    bool ok = mine[0] >= 0 && mine[1] >= 0 && theirs[0] >= 0 && theirs[1] >= 0;
                    for (size_t k = 0; ok && k < board.size(); ++k) {
                        bd[k] = parseCard(board[k].c_str());
                        ok = bd[k] >= 0;
                    }
                    if (ok) {
                        double v = adj;
                        if (allinAdjustedWinnings(action, mine, theirs, bd, int(clientPos),
                                                  winnings, v)) {
                            adj = v;
                            didAdjust = true;
                            ++res.allinAdjusted;
                        }
                    }
                }
                sumAdj += adj;
                sumAdjSq += adj * adj;
                if (log != nullptr) {
                    long long wonPot = 0;
                    jsonInt(resp, "won_pot", wonPot);
                    auto joinCards = [](const std::vector<std::string>& v) {
                        std::string s;
                        for (size_t k = 0; k < v.size(); ++k) {
                            if (k) s += ' ';
                            s += v[k];
                        }
                        return s;
                    };
                    std::fprintf(log, "%d,%lld,%lld,%.1f,%lld,%d,\"%s\",\"%s\",\"%s\",\"%s\"\n", h,
                                 clientPos, winnings, adj, wonPot < 0 ? -wonPot : wonPot,
                                 didAdjust ? 1 : 0, action.c_str(), joinCards(hole).c_str(),
                                 joinCards(botHole).c_str(), joinCards(board).c_str());
                    // 逐手落盘：分析工具要能实时读，且崩溃/中断不丢历史。
                    std::fflush(log);
                }
                if (winnings > 0) ++res.wins;
                else if (winnings < 0) ++res.losses;
                else ++res.pushes;
                finished = true;
                if (h % 20 == 0 || h == cfg.hands) {
                    std::printf("  %4d 手: 本手 %+6lld | 累计 %+8lld (%+.1f 筹码/手)\n", h, winnings,
                                res.winnings, double(res.winnings) / double(res.hands));
                    std::fflush(stdout);
                }
                break;
            }

            if (hole.size() < 2) {
                ++res.errors;
                std::printf("响应缺少 hole_cards: %s\n", resp.c_str());
                token.clear();
                break;
            }

            HUState st;
            if (!buildSlumbotState(action, hole[0], hole[1], board, int(clientPos), rng, st)) {
                ++res.errors;
                std::printf("无法重建状态: action=%s board=%zu\n", action.c_str(), board.size());
                token.clear();
                break;
            }
            if (st.toAct != 1 - int(clientPos)) {
                ++res.errors;
                std::printf("行动权不一致: toAct=%d clientPos=%lld action=%s\n", int(st.toAct),
                            clientPos, action.c_str());
                token.clear();
                break;
            }

            Move mv;
            if (cfg.policy == "fold" || cfg.policy == "random") {
                LegalMove lm[kNumActions];
                const int n = legalMoves(st, lm);
                if (n <= 0) {
                    ++res.errors;
                    token.clear();
                    break;
                }
                int pick = 0;
                if (cfg.policy == "fold") {
                    // 能过牌就过，否则弃牌
                    pick = 0;
                    for (int i = 0; i < n; ++i) {
                        if (lm[i].slot == kSlotCheckCall) {
                            pick = i;
                            break;
                        }
                    }
                    for (int i = 0; i < n; ++i) {
                        if (lm[i].slot == kSlotFold) {
                            pick = i;
                            break;
                        }
                    }
                } else {
                    pick = static_cast<int>(rng() % static_cast<size_t>(n));
                }
                mv = lm[pick].mv;
                if (mv.kind == MK_RAISE && lm[pick].toHi > int(mv.to)) {
                    std::uniform_real_distribution<double> u01(0.0, 1.0);
                    mv.to = static_cast<int16_t>(amountInBand(int(mv.to), lm[pick].toHi, u01(rng)));
                }
            } else {
                // rangeEq：与训练同一口径。训练时每个决策都真算并喂给范围残差
                // 支路，部署若固定喂 0.5，同一份权重看到的输入就和训练不同。
                float rq = 0.5f;
                if (cfg.rangeEqSamples > 0) {
                    RangeObs last;
                    if (slumbotLastOppDecision(action, hole[0], hole[1], board, int(clientPos),
                                               rng, last)) {
                        rq = rangeEquityAt(st, 1 - int(clientPos), last,
                                           &actors[int(clientPos)], cfg.rangeEqSamples,
                                           cfg.rangeEqIters, rng);
                        ++res.rangeEqDecisions;
                    }
                }
                mv = policyMove(&actors[0], &actors[1], st, rng, cfg.temp, rq);
                // 河牌大注剥削：无人下注且底池够大时，直接按 1.5x 池下注。
                // 这是外部范围诊断给出的最高 EV 动作（第 87 条）。
                if (cfg.riverOverbetPot > 0.0f && st.street == ST_RIVER &&
                    st.boardN == 5 && toCall(st, st.toAct) == 0 &&
                    st.pot() >= cfg.riverOverbetMinPot && st.stack[st.toAct] > 0) {
                    const int to = int(st.contrib[st.toAct]) +
                                   int(std::lround(double(st.pot()) * cfg.riverOverbetPot));
                    mv = makeRaise(st, to);
                    ++res.riverOverbets;
                }
                // 3-bet 放大：只动 numRaises==1 的再加注；向上钳到筹码上限。
                if (cfg.preflop3betMult > 1.0f && st.street == ST_PREFLOP &&
                    st.numRaises == 1 && (mv.kind == MK_RAISE || mv.kind == MK_ALLIN)) {
                    const int cap = int(st.contrib[st.toAct]) + int(st.stack[st.toAct]);
                    int to = int(std::lround(double(mv.to) * double(cfg.preflop3betMult)));
                    if (to > cap) to = cap;
                    if (to > int(mv.to)) mv.to = static_cast<int16_t>(to);
                }
                if (cfg.postflopBetMult > 1.0f && st.street > ST_PREFLOP &&
                    (mv.kind == MK_RAISE || mv.kind == MK_ALLIN)) {
                    const int cap = int(st.contrib[st.toAct]) + int(st.stack[st.toAct]);
                    int to = int(std::lround(double(mv.to) * double(cfg.postflopBetMult)));
                    if (to > cap) to = cap;
                    if (to > int(mv.to)) mv.to = static_cast<int16_t>(to);
                }
            }
            // 翻前进攻上限：弱牌不在大池里继续加注（4-bet/5-bet 改成跟注）。
            if (cfg.preflopAggroPct > 0.0f && st.street == ST_PREFLOP &&
                st.numRaises >= cfg.preflopAggroMaxRaises &&
                (mv.kind == MK_RAISE || mv.kind == MK_ALLIN)) {
                int hole[2] = {int(st.hole[st.toAct][0]), int(st.hole[st.toAct][1])};
                int bd0[5] = {0, 0, 0, 0, 0};
                if (handStrength(hole, bd0, 0) < cfg.preflopAggroPct) {
                    mv = Move{};
                    mv.kind = MK_CALL;
                    mv.to = st.contrib[1 - st.toAct];
                    ++res.preflopAggroCapped;
                }
            }
            // 翻牌进攻上限：中等牌不在翻牌继续加注（改成跟注）。
            if (cfg.flopAggroPct > 0.0f && st.street == ST_FLOP &&
                (mv.kind == MK_RAISE || mv.kind == MK_ALLIN)) {
                int hole[2] = {int(st.hole[st.toAct][0]), int(st.hole[st.toAct][1])};
                int bd[5] = {0, 0, 0, 0, 0};
                for (int k = 0; k < int(st.boardN); ++k) bd[k] = int(st.board[k]);
                if (handStrength(hole, bd, int(st.boardN)) < cfg.flopAggroPct) {
                mv = Move{};
                mv.kind = MK_CALL;
                mv.to = st.contrib[1 - st.toAct];
                    ++res.flopAggroCapped;
                }
            }
            // 全下纪律 guard：只保留顶范围跟注（部署剥削，不改变 actor）。
            if ((cfg.allinGuardPrePct > 0.0f || cfg.allinGuardPostCat >= 0) &&
                mv.kind != MK_FOLD &&
                allinGuardFoldImpl(st, cfg.allinGuardPrePct, cfg.allinGuardPostCat,
                                   cfg.allinGuardMinRisk)) {
                mv = Move{};
                mv.kind = MK_FOLD;
                mv.to = st.contrib[st.toAct];
                ++res.allinGuardFolds;
            }
            const std::string incr = moveToIncr(mv);
            const std::string body2 = "{\"token\": \"" + token + "\", \"incr\": \"" + incr + "\"}";
            if (!postJson(cfg.host, "/slumbot/api/act", body2, resp, 2, why)) {
                ++res.errors;
                std::printf("act 失败: incr=%s action=%s (%s)\n", incr.c_str(), action.c_str(),
                            why.c_str());
                std::fflush(stdout);
                token.clear();
                break;
            }
            if (cfg.delayMs > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(cfg.delayMs));
            }
        }
    }
    if (log != nullptr) std::fclose(log);
    if (res.hands > 1) {
        const double n = double(res.hands);
        const double mean = sumW / n;
        const double var = std::max(sumSq / n - mean * mean, 0.0);
        res.chipsStderr = std::sqrt(var / n);
        const double meanAdj = sumAdj / n;
        const double varAdj = std::max(sumAdjSq / n - meanAdj * meanAdj, 0.0);
        res.adjustedWinnings = sumAdj;
        res.adjustedStderr = std::sqrt(varAdj / n);
    }
    return res;
}

}  // namespace texas
