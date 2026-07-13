static long long ds4_deepspec_ds13b_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (long long)tv.tv_sec * 1000LL + tv.tv_usec / 1000;
}

static void generate_deepspec_generate_dflash_job(server *s, job *j) {
    const char *base = getenv("DS4_DEEPSPEC_DUMP_PREFIX");
    if (!base || !base[0]) base = "/tmp/ds4_deepspec_dflash";

    static uint64_t counter = 0;

    char err[1024];
    err[0] = 0;

    char *sidecar_url = ds4_json_parse_optional_string_field(j->req.raw_body, "sidecar_url");
    if (!sidecar_url) {
        const char *env_url = getenv("DS4_DEEPSPEC_SIDECAR_URL");
        sidecar_url = xstrdup(env_url && env_url[0]
            ? env_url
            : "http://127.0.0.1:8091/v1/ds4/dflash_propose_from_raw");
    }

    double temperature = ds4_json_parse_optional_double(j->req.raw_body, "temperature", 0.0);
    if (!isfinite(temperature) || temperature < 0.0) temperature = 0.0;

    int top_k = j->req.spec_top_k > 0 ? j->req.spec_top_k : 8;
    if (top_k > 64) top_k = 64;

    int requested_max_tokens = j->req.max_tokens > 0 ? j->req.max_tokens : 1;
    if (requested_max_tokens > 256) requested_max_tokens = 256;

    int chat_stop_token = -1;
    ds4_tokens chat_stop_tokens = {0};
    ds4_tokenize_rendered_chat(s->engine, "<｜end▁of▁sentence｜>", &chat_stop_tokens);
    if (chat_stop_tokens.len == 1) chat_stop_token = chat_stop_tokens.v[0];
    ds4_tokens_free(&chat_stop_tokens);

    int initial_max_draft_tokens = requested_max_tokens < 7 ? requested_max_tokens : 7;
    if (initial_max_draft_tokens < 1) initial_max_draft_tokens = 1;

    int all_committed[256];
    int cycle_draft_counts[256];
    int cycle_committed_counts[256];
    int cycle_accepted_counts[256];
    int cycle_positions[256];

    int total_committed = 0;
    int total_accepted = 0;
    int total_draft_tokens = 0;
    int total_tried = 0;
    int cycles = 0;

    int last_sidecar_status = 0;
    int last_draft_token_count = 0;
    int last_max_draft_tokens = 0;
    char proposal_mode[64] = "unknown";
    char checkpoint_sha256[80] = "unknown";
    char vocab_sha256[80] = "unknown";
    long long ds13b_endpoint_start_ms = ds4_deepspec_ds13b_now_ms();
    int runtime_prompt_token_count = ds4_session_pos(s->session);
    long long ds13b_first_token_ms = -1;
    long long ds13b_sidecar_total_ms = 0;
    long long ds13b_cycle_total_ms[256] = {0};
    long long ds13b_cycle_sidecar_ms[256] = {0};
    long long ds13b_cycle_verifier_ms[256] = {0};
    long long ds13b_target_probs_total_ms = 0;
    long long ds13b_draft_probs_load_total_ms = 0;
    long long ds13b_verify_residual_total_ms = 0;
    long long ds13b_decode_text_ms = 0;
    bool residual_any = false;
    bool eos_seen = false;
    char *last_draft_probs_file = NULL;

    bool debug_dflash_verify = false;
    const char *debug_dflash_env = getenv("DS4_DEEPSPEC_DFLASH_DEBUG");
    if (debug_dflash_env && debug_dflash_env[0] && strcmp(debug_dflash_env, "0")) debug_dflash_verify = true;

    int debug_dflash_tried_n = 0;
    int debug_dflash_draft_token_ids[256];
    int debug_dflash_verified_token_ids[256];
    int debug_dflash_target_argmax_token_ids[256];
    double debug_dflash_target_argmax_probs[256];
    int debug_dflash_draft_matches_target_argmax[256];
    double debug_dflash_draft_prob_of_target_argmax[256];
    int debug_dflash_draft_rank_of_target_argmax[256];
    int debug_dflash_draft_top1_token_ids[256];
    double debug_dflash_draft_top1_probs[256];
    double debug_dflash_draft_token_probs[256];
    double debug_dflash_target_token_probs[256];
    double debug_dflash_accept_probs[256];
    double debug_dflash_accept_randoms[256];
    double debug_dflash_residual_sums[256];
    int debug_dflash_fallback_token_ids[256];

    int debug_dflash_proposal_n = 0;
    int debug_dflash_proposal_cycle_ids[1024];
    int debug_dflash_proposal_token_ids[1024];
    double debug_dflash_proposal_token_probs[1024];

    while (total_committed < requested_max_tokens && cycles < requested_max_tokens && !eos_seen) {
        long long ds13b_cycle_t0_ms = ds4_deepspec_ds13b_now_ms();
        int ds13b_cycle_index = cycles;
        counter++;

        char prefix[4096];
        int n = snprintf(prefix, sizeof(prefix), "%s.generate.%llu",
                         base, (unsigned long long)counter);
        if (n <= 0 || (size_t)n >= sizeof(prefix)) {
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, "dFlash dump prefix too long");
            return;
        }

        err[0] = 0;
        if (ds4_session_dump_deepspec_sequence_raw(s->session, prefix, err, sizeof(err)) != 0) {
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, err[0] ? err : "dFlash dump failed");
            return;
        }

        char meta_path[4096], input_path[4096], hidden_path[4096], last_path[4096];
        int meta_n = snprintf(meta_path, sizeof(meta_path), "%s.meta.json", prefix);
        int input_n = snprintf(input_path, sizeof(input_path), "%s.input_ids.i32", prefix);
        int hidden_n = snprintf(hidden_path, sizeof(hidden_path), "%s.target_hidden.f32", prefix);
        int last_n = snprintf(last_path, sizeof(last_path), "%s.target_last_hidden.f32", prefix);
        if (meta_n <= 0 || input_n <= 0 || hidden_n <= 0 || last_n <= 0 ||
            (size_t)meta_n >= sizeof(meta_path) ||
            (size_t)input_n >= sizeof(input_path) ||
            (size_t)hidden_n >= sizeof(hidden_path) ||
            (size_t)last_n >= sizeof(last_path)) {
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, "dFlash dump path too long");
            return;
        }

        int remaining = requested_max_tokens - total_committed;
        int max_draft_tokens = remaining < 7 ? remaining : 7;
        if (max_draft_tokens < 1) max_draft_tokens = 1;
        last_max_draft_tokens = max_draft_tokens;

        buf payload = {0};
        buf_puts(&payload, "{\"meta\":");
        json_escape(&payload, meta_path);
        buf_puts(&payload, ",\"input_ids\":");
        json_escape(&payload, input_path);
        buf_puts(&payload, ",\"target_hidden_states\":");
        json_escape(&payload, hidden_path);
        buf_puts(&payload, ",\"target_last_hidden_states\":");
        json_escape(&payload, last_path);
        buf_printf(&payload, ",\"top_k\":%d,\"max_draft_tokens\":%d,\"temperature\":%.17g,\"write_draft_probs\":true}", top_k, max_draft_tokens, temperature);

        int sidecar_status = 0;
        err[0] = 0;
        long long ds13b_sidecar_t0_ms = ds4_deepspec_ds13b_now_ms();
        char *sidecar_body = ds4_deepspec_http_post_json(sidecar_url, payload.ptr ? payload.ptr : "{}",
                                                         &sidecar_status, err, sizeof(err));
        long long ds13b_sidecar_dt_ms = ds4_deepspec_ds13b_now_ms() - ds13b_sidecar_t0_ms;
        ds13b_sidecar_total_ms += ds13b_sidecar_dt_ms;
        if (ds13b_cycle_index >= 0 && ds13b_cycle_index < 256) ds13b_cycle_sidecar_ms[ds13b_cycle_index] = ds13b_sidecar_dt_ms;
        buf_free(&payload);
        last_sidecar_status = sidecar_status;

        if (!sidecar_body) {
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, err[0] ? err : "dFlash sidecar request failed");
            return;
        }

        if (sidecar_status != 200) {
            buf body = {0};
            buf_puts(&body, "{\"object\":\"ds4.deepspec_generate_dflash\",\"mode\":\"loop\",\"ok\":false,\"error\":\"sidecar returned non-200\",\"sidecar_http_status\":");
            buf_printf(&body, "%d", sidecar_status);
            buf_puts(&body, ",\"cycles\":");
            buf_printf(&body, "%d", cycles);
            buf_puts(&body, ",\"committed_tokens\":");
            buf_printf(&body, "%d", total_committed);
            buf_puts(&body, ",\"sidecar_response\":");
            if (sidecar_body[0] == '{') buf_puts(&body, sidecar_body); else json_escape(&body, sidecar_body);
            buf_putc(&body, 125);
            http_response(j->fd, s->enable_cors, 502, "application/json", body.ptr);
            buf_free(&body);
            free(sidecar_body);
            free(last_draft_probs_file);
            free(sidecar_url);
            return;
        }

        char *sidecar_proposal_mode = ds4_json_parse_optional_string_dup(sidecar_body, "proposal_mode");
        char *sidecar_checkpoint_sha256 = ds4_json_parse_optional_string_dup(sidecar_body, "checkpoint_sha256");
        char *sidecar_vocab_sha256 = ds4_json_parse_optional_string_dup(sidecar_body, "vocab_sha256");
        const bool sidecar_identity_changed =
            (sidecar_proposal_mode && strcmp(proposal_mode, "unknown") && strcmp(proposal_mode, sidecar_proposal_mode)) ||
            (sidecar_checkpoint_sha256 && strcmp(checkpoint_sha256, "unknown") && strcmp(checkpoint_sha256, sidecar_checkpoint_sha256)) ||
            (sidecar_vocab_sha256 && strcmp(vocab_sha256, "unknown") && strcmp(vocab_sha256, sidecar_vocab_sha256));
        if (sidecar_identity_changed) {
            free(sidecar_proposal_mode);
            free(sidecar_checkpoint_sha256);
            free(sidecar_vocab_sha256);
            free(sidecar_body);
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 409, "sidecar mode/checkpoint/vocab changed during request");
            return;
        }
        if (sidecar_proposal_mode) snprintf(proposal_mode, sizeof(proposal_mode), "%s", sidecar_proposal_mode);
        if (sidecar_checkpoint_sha256) snprintf(checkpoint_sha256, sizeof(checkpoint_sha256), "%s", sidecar_checkpoint_sha256);
        if (sidecar_vocab_sha256) snprintf(vocab_sha256, sizeof(vocab_sha256), "%s", sidecar_vocab_sha256);
        free(sidecar_proposal_mode);
        free(sidecar_checkpoint_sha256);
        free(sidecar_vocab_sha256);

        int draft_tokens[16];
        int n_draft = 0;
        if (!ds4_json_parse_int_array16(sidecar_body, "draft_token_ids", draft_tokens, &n_draft, 16, err, sizeof(err))) {
            free(sidecar_body);
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, err[0] ? err : "sidecar missing draft_token_ids");
            return;
        }

        if (n_draft <= 0) {
            free(sidecar_body);
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, "sidecar returned empty draft");
            return;
        }

        last_draft_token_count = n_draft;

        double draft_token_probs[16];
        int n_probs = 0;
        if (!ds4_json_parse_float_array16(sidecar_body, "draft_token_probs", draft_token_probs, &n_probs, 16, err, sizeof(err)) || n_probs != n_draft) {
            free(sidecar_body);
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, "sidecar missing draft_token_probs");
            return;
        }

        char *draft_probs_file = ds4_json_parse_optional_string_dup(sidecar_body, "draft_probs_file");

        if (debug_dflash_verify) {
            for (int pi = 0; pi < n_draft && debug_dflash_proposal_n < 1024; pi++) {
                debug_dflash_proposal_cycle_ids[debug_dflash_proposal_n] = cycles;
                debug_dflash_proposal_token_ids[debug_dflash_proposal_n] = draft_tokens[pi];
                debug_dflash_proposal_token_probs[debug_dflash_proposal_n] = draft_token_probs[pi];
                debug_dflash_proposal_n++;
            }
        }
        bool residual_enabled = draft_probs_file && draft_probs_file[0];
        if (residual_enabled) residual_any = true;

        int remaining_commit = requested_max_tokens - total_committed;
        if (remaining_commit < 1) remaining_commit = 1;

        ds4_dflash_verify_result vr;
        ds4_dflash_verify_result_init(&vr);

        long long ds13b_verify_residual_t0_ms = ds4_deepspec_ds13b_now_ms();
        const bool correctness_first_proposal =
            !strcmp(proposal_mode, "target_assisted_correctness_first");
        if (!ds4_dflash_verify_and_commit(s,
                                          draft_tokens,
                                          draft_token_probs,
                                          n_draft,
                                          draft_probs_file,
                                          temperature,
                                          remaining_commit,
                                          !correctness_first_proposal,
                                          &vr,
                                          err,
                                          sizeof(err))) {
            free(draft_probs_file);
            free(sidecar_body);
            free(last_draft_probs_file);
            free(sidecar_url);
            http_error(j->fd, s->enable_cors, 500, err[0] ? err : "dflash verify helper failed");
            return;
        }
        ds13b_verify_residual_total_ms += ds4_deepspec_ds13b_now_ms() - ds13b_verify_residual_t0_ms;
        if (ds13b_cycle_index >= 0 && ds13b_cycle_index < 256) {
            ds13b_cycle_verifier_ms[ds13b_cycle_index] =
                ds4_deepspec_ds13b_now_ms() - ds13b_verify_residual_t0_ms;
        }
        ds13b_target_probs_total_ms += vr.timing_target_probs_ms;
        ds13b_draft_probs_load_total_ms += vr.timing_draft_probs_load_ms;

        int committed_n = vr.committed_n;
        int accepted_n = vr.accepted_n;
        int tried_n = vr.tried_n;

        if (debug_dflash_verify) {
            for (int vi = 0; vi < tried_n && debug_dflash_tried_n < 256; vi++) {
                debug_dflash_draft_token_ids[debug_dflash_tried_n] = draft_tokens[vi];
                debug_dflash_verified_token_ids[debug_dflash_tried_n] = vr.verified_tokens[vi];
                debug_dflash_target_argmax_token_ids[debug_dflash_tried_n] = vr.target_argmax_tokens[vi];
                debug_dflash_target_argmax_probs[debug_dflash_tried_n] = vr.target_argmax_probs[vi];
                debug_dflash_draft_matches_target_argmax[debug_dflash_tried_n] = vr.draft_matches_target_argmax[vi];
                debug_dflash_draft_prob_of_target_argmax[debug_dflash_tried_n] = vr.draft_prob_of_target_argmax[vi];
                debug_dflash_draft_rank_of_target_argmax[debug_dflash_tried_n] = vr.draft_rank_of_target_argmax[vi];
                debug_dflash_draft_top1_token_ids[debug_dflash_tried_n] = vr.draft_top1_token_ids[vi];
                debug_dflash_draft_top1_probs[debug_dflash_tried_n] = vr.draft_top1_probs[vi];
                debug_dflash_draft_token_probs[debug_dflash_tried_n] = draft_token_probs[vi];
                debug_dflash_target_token_probs[debug_dflash_tried_n] = vr.target_token_probs[vi];
                debug_dflash_accept_probs[debug_dflash_tried_n] = vr.accept_probs[vi];
                debug_dflash_accept_randoms[debug_dflash_tried_n] = vr.accept_randoms[vi];
                debug_dflash_residual_sums[debug_dflash_tried_n] = vr.residual_sums[vi];
                debug_dflash_fallback_token_ids[debug_dflash_tried_n] = vr.fallback_tokens[vi];
                debug_dflash_tried_n++;
            }
        }

        int visible_committed_n = 0;
        for (int ci = 0; ci < committed_n &&
                         (total_committed + visible_committed_n) < requested_max_tokens; ci++) {
            if (vr.committed[ci] == ds4_token_eos(s->engine)) {
                eos_seen = true;
                continue;
            }
            all_committed[total_committed + visible_committed_n++] = vr.committed[ci];
            size_t committed_piece_len = 0;
            char *committed_piece = ds4_token_text(s->engine, vr.committed[ci], &committed_piece_len);
            const char *chat_stop_text = "<｜end▁of▁sentence｜>";
            const bool is_chat_stop = committed_piece &&
                committed_piece_len == strlen(chat_stop_text) &&
                !memcmp(committed_piece, chat_stop_text, committed_piece_len);
            free(committed_piece);
            if (vr.committed[ci] == chat_stop_token || is_chat_stop) eos_seen = true;
        }

        free(last_draft_probs_file);
        last_draft_probs_file = draft_probs_file;
        draft_probs_file = NULL;

        free(sidecar_body);

        cycle_draft_counts[cycles] = n_draft;
        cycle_committed_counts[cycles] = visible_committed_n;
        cycle_accepted_counts[cycles] = accepted_n;
        cycle_positions[cycles] = ds4_session_pos(s->session);

        cycles++;
        total_committed += visible_committed_n;
        if (ds13b_first_token_ms < 0 && total_committed > 0) ds13b_first_token_ms = ds4_deepspec_ds13b_now_ms() - ds13b_endpoint_start_ms;
        total_accepted += accepted_n;
        total_draft_tokens += n_draft;
        if (ds13b_cycle_index >= 0 && ds13b_cycle_index < 256) ds13b_cycle_total_ms[ds13b_cycle_index] = ds4_deepspec_ds13b_now_ms() - ds13b_cycle_t0_ms;
        total_tried += tried_n;

        if (committed_n <= 0) break;
    }

    buf decoded = {0};
    long long ds13b_decode_t0_ms = ds4_deepspec_ds13b_now_ms();
    for (int di = 0; di < total_committed; di++) {
        int tok = all_committed[di];
        if (tok == ds4_token_eos(s->engine)) continue;
        size_t piece_len = 0;
        char *piece = ds4_token_text(s->engine, tok, &piece_len);
        if (piece) {
            buf_append(&decoded, piece, piece_len);
            free(piece);
        }
    }
    ds13b_decode_text_ms = ds4_deepspec_ds13b_now_ms() - ds13b_decode_t0_ms;

    buf body = {0};
    buf_puts(&body, "{\"object\":\"ds4.deepspec_generate_dflash\",\"mode\":\"loop\",");
    buf_puts(&body, "\"requested_max_tokens\":");
    buf_printf(&body, "%d", requested_max_tokens);
    buf_puts(&body, ",\"initial_max_draft_tokens\":");
    buf_printf(&body, "%d", initial_max_draft_tokens);
    buf_puts(&body, ",\"last_max_draft_tokens\":");
    buf_printf(&body, "%d", last_max_draft_tokens);
    buf_puts(&body, ",\"cycles\":");
    buf_printf(&body, "%d", cycles);
    buf_puts(&body, ",\"sidecar_url\":");
    json_escape(&body, sidecar_url);
    buf_puts(&body, ",\"sidecar_http_status\":");
    buf_printf(&body, "%d", last_sidecar_status);
    buf_puts(&body, ",\"proposal_mode\":");
    json_escape(&body, proposal_mode);
    buf_puts(&body, ",\"checkpoint_sha256\":");
    json_escape(&body, checkpoint_sha256);
    buf_puts(&body, ",\"vocab_sha256\":");
    json_escape(&body, vocab_sha256);
    buf_puts(&body, ",\"draft_token_count\":");
    buf_printf(&body, "%d", last_draft_token_count);
    buf_puts(&body, ",\"total_draft_token_count\":");
    buf_printf(&body, "%d", total_draft_tokens);
    buf_puts(&body, ",\"cycle_draft_token_counts\":");
    ds4_append_int_array_json(&body, cycle_draft_counts, cycles);
    buf_puts(&body, ",\"cycle_committed_tokens\":");
    ds4_append_int_array_json(&body, cycle_committed_counts, cycles);
    buf_puts(&body, ",\"cycle_accepted_draft_tokens\":");
    ds4_append_int_array_json(&body, cycle_accepted_counts, cycles);
    buf_puts(&body, ",\"cycle_positions\":");
    ds4_append_int_array_json(&body, cycle_positions, cycles);
    buf_puts(&body, ",\"accepted_draft_tokens\":");
    buf_printf(&body, "%d", total_accepted);
    buf_puts(&body, ",\"committed_tokens\":");
    buf_printf(&body, "%d", total_committed);
    buf_puts(&body, ",\"committed_token_ids\":");
    ds4_append_int_array_json(&body, all_committed, total_committed);
    if (j->req.ds4_return_runtime_metrics || j->req.ds4_return_token_ids) {
        long long runtime_now_ms = ds4_deepspec_ds13b_now_ms();
        long long runtime_total_ms = runtime_now_ms - ds13b_endpoint_start_ms;
        long long runtime_decode_ms = ds13b_first_token_ms >= 0 ?
            runtime_total_ms - ds13b_first_token_ms : 0;
        if (runtime_decode_ms < 0) runtime_decode_ms = 0;
        double runtime_tps = total_committed > 1 && runtime_decode_ms > 0 ?
            (double)(total_committed - 1) * 1000.0 / (double)runtime_decode_ms : 0.0;
        buf_puts(&body, ",\"ds4_runtime\":{\"schema\":\"ds4_runtime_generation_v1\","
                         "\"mode\":\"speculative\",\"token_id_provenance\":\"native_commit_path\","
                         "\"completion_token_ids\":");
        ds4_append_int_array_json(&body, all_committed, total_committed);
        buf_puts(&body, ",\"completion_token_count\":"); buf_printf(&body, "%d", total_committed);
        buf_puts(&body, ",\"prompt_token_count\":"); buf_printf(&body, "%d", runtime_prompt_token_count);
        buf_puts(&body, ",\"finish_reason\":"); json_escape(&body, eos_seen ? "stop" : "length");
        buf_puts(&body, ",\"eos_token_id\":"); buf_printf(&body, "%d", ds4_token_eos(s->engine));
        buf_puts(&body, ",\"stop_reason\":"); json_escape(&body, eos_seen ? "eos" : "max_tokens");
        buf_puts(&body, ",\"prompt_processing_ms\":0.0,\"ttft_ms\":");
        buf_printf(&body, "%lld", ds13b_first_token_ms >= 0 ? ds13b_first_token_ms : 0);
        buf_puts(&body, ",\"decode_ms\":"); buf_printf(&body, "%lld", runtime_decode_ms);
        buf_puts(&body, ",\"total_generation_ms\":"); buf_printf(&body, "%lld", runtime_total_ms);
        buf_puts(&body, ",\"generation_tokens_per_second\":"); buf_printf(&body, "%.6f", runtime_tps);
        buf_puts(&body, ",\"proposed_draft_tokens\":"); buf_printf(&body, "%d", total_draft_tokens);
        buf_puts(&body, ",\"accepted_draft_tokens\":"); buf_printf(&body, "%d", total_accepted);
        buf_puts(&body, ",\"accepted_tokens_per_block\":");
        buf_printf(&body, "%.6f", cycles > 0 ? (double)total_accepted / (double)cycles : 0.0);
        buf_puts(&body, ",\"cycles\":"); buf_printf(&body, "%d", cycles);
        buf_puts(&body, ",\"fallback_reason\":null}");
    }
    buf_puts(&body, ",\"text\":");
    json_escape(&body, decoded.ptr ? decoded.ptr : "");
    buf_puts(&body, ",\"content\":");
    json_escape(&body, decoded.ptr ? decoded.ptr : "");
    buf_puts(&body, ",\"position\":");
    buf_printf(&body, "%d", ds4_session_pos(s->session));
    buf_puts(&body, ",\"residual_sampling\":");
    buf_puts(&body, residual_any ? "true" : "false");
    buf_puts(&body, ",\"eos\":");
    buf_puts(&body, eos_seen ? "true" : "false");
    buf_puts(&body, ",\"total_tried_tokens\":");
    buf_printf(&body, "%d", total_tried);
    if (debug_dflash_verify) {
    buf_puts(&body, ",\"proposal_debug\":{");
    buf_puts(&body, "\"proposed_tokens\":");
    buf_printf(&body, "%d", debug_dflash_proposal_n);
    buf_puts(&body, ",\"cycle_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_proposal_cycle_ids, debug_dflash_proposal_n);
    buf_puts(&body, ",\"token_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_proposal_token_ids, debug_dflash_proposal_n);
    buf_puts(&body, ",\"token_probs\":");
    ds4_append_double_array_json(&body, debug_dflash_proposal_token_probs, debug_dflash_proposal_n);
    buf_puts(&body, "}");
    }

    if (debug_dflash_verify) {
    buf_puts(&body, ",\"verify_debug\":{");
    buf_puts(&body, "\"tried_tokens\":");
    buf_printf(&body, "%d", debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_token_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_draft_token_ids, debug_dflash_tried_n);
    buf_puts(&body, ",\"verified_token_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_verified_token_ids, debug_dflash_tried_n);
    buf_puts(&body, ",\"target_argmax_token_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_target_argmax_token_ids, debug_dflash_tried_n);
    buf_puts(&body, ",\"target_argmax_probs\":");
    ds4_append_double_array_json(&body, debug_dflash_target_argmax_probs, debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_matches_target_argmax\":");
    ds4_append_int_array_json(&body, debug_dflash_draft_matches_target_argmax, debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_prob_of_target_argmax\":");
    ds4_append_double_array_json(&body, debug_dflash_draft_prob_of_target_argmax, debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_rank_of_target_argmax\":");
    ds4_append_int_array_json(&body, debug_dflash_draft_rank_of_target_argmax, debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_top1_token_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_draft_top1_token_ids, debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_top1_probs\":");
    ds4_append_double_array_json(&body, debug_dflash_draft_top1_probs, debug_dflash_tried_n);
    buf_puts(&body, ",\"draft_token_probs\":");
    ds4_append_double_array_json(&body, debug_dflash_draft_token_probs, debug_dflash_tried_n);
    buf_puts(&body, ",\"target_token_probs\":");
    ds4_append_double_array_json(&body, debug_dflash_target_token_probs, debug_dflash_tried_n);
    buf_puts(&body, ",\"accept_probs\":");
    ds4_append_double_array_json(&body, debug_dflash_accept_probs, debug_dflash_tried_n);
    buf_puts(&body, ",\"accept_randoms\":");
    ds4_append_double_array_json(&body, debug_dflash_accept_randoms, debug_dflash_tried_n);
    buf_puts(&body, ",\"residual_sums\":");
    ds4_append_double_array_json(&body, debug_dflash_residual_sums, debug_dflash_tried_n);
    buf_puts(&body, ",\"fallback_token_ids\":");
    ds4_append_int_array_json(&body, debug_dflash_fallback_token_ids, debug_dflash_tried_n);
    buf_puts(&body, "}");
    }

    buf_puts(&body, ",\"draft_probs_file\":");
    if (last_draft_probs_file && last_draft_probs_file[0]) json_escape(&body, last_draft_probs_file);
    else buf_puts(&body, "null");
    long long ds13b_endpoint_total_ms = ds4_deepspec_ds13b_now_ms() - ds13b_endpoint_start_ms;
    buf_puts(&body, ",\"timing_ms\":{");
    buf_puts(&body, "\"endpoint_total\":");
    buf_printf(&body, "%lld", ds13b_endpoint_total_ms);
    buf_puts(&body, ",\"first_token\":");
    if (ds13b_first_token_ms >= 0) buf_printf(&body, "%lld", ds13b_first_token_ms); else buf_puts(&body, "null");
    buf_puts(&body, ",\"sidecar_total\":");
    buf_printf(&body, "%lld", ds13b_sidecar_total_ms);
    buf_puts(&body, ",\"sidecar_mean\":");
    buf_printf(&body, "%.3f", cycles > 0 ? ((double)ds13b_sidecar_total_ms / (double)cycles) : 0.0);
    long long ds13b_non_sidecar_total_ms = ds13b_endpoint_total_ms - ds13b_sidecar_total_ms;
    if (ds13b_non_sidecar_total_ms < 0) ds13b_non_sidecar_total_ms = 0;
    buf_puts(&body, ",\"target_probs_total\":");
    buf_printf(&body, "%lld", ds13b_target_probs_total_ms);
    buf_puts(&body, ",\"draft_probs_load_total\":");
    buf_printf(&body, "%lld", ds13b_draft_probs_load_total_ms);
    buf_puts(&body, ",\"verify_residual_total\":");
    buf_printf(&body, "%lld", ds13b_verify_residual_total_ms);
    buf_puts(&body, ",\"decode_text\":");
    buf_printf(&body, "%lld", ds13b_decode_text_ms);
    buf_puts(&body, ",\"non_sidecar_total_est\":");
    buf_printf(&body, "%lld", ds13b_non_sidecar_total_ms);
    buf_puts(&body, ",\"cycle_total\":[");
    for (int ti = 0; ti < cycles; ti++) { if (ti) buf_putc(&body, 44); buf_printf(&body, "%lld", ds13b_cycle_total_ms[ti]); }
    buf_puts(&body, "],\"cycle_sidecar\":[");
    for (int ti = 0; ti < cycles; ti++) { if (ti) buf_putc(&body, 44); buf_printf(&body, "%lld", ds13b_cycle_sidecar_ms[ti]); }
    buf_puts(&body, "],\"cycle_verifier\":[");
    for (int ti = 0; ti < cycles; ti++) { if (ti) buf_putc(&body, 44); buf_printf(&body, "%lld", ds13b_cycle_verifier_ms[ti]); }
    buf_puts(&body, "]}");
    buf_puts(&body, ",\"finish_reason\":");
    json_escape(&body, eos_seen ? "stop" : "length");
    buf_puts(&body, ",\"usage\":{\"completion_tokens\":");
    buf_printf(&body, "%d", total_committed);
    buf_putc(&body, 125);
    buf_puts(&body, ",\"ok\":true}");

    http_response(j->fd, s->enable_cors, 200, "application/json", body.ptr);

    buf_free(&body);
    buf_free(&decoded);
    free(last_draft_probs_file);
    free(sidecar_url);
}
