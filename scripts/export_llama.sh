#!/usr/bin/env bash
# Export Llama 3.2 1B Instruct to an ExecuTorch .pte (XNNPACK, 8-bit dynamic act / 4-bit weights).
# usage: scripts/export_llama.sh GROUP_SIZE EMBED_QUANT(none|8,0|4,32) MAX_SEQ OUT.pte
# env:   ET_ROOT=path to an executorch checkout (pinned tag, see README)
#        LLAMA_DIR=dir with consolidated.00.pth, params.json, tokenizer.model
#        (Meta "original" checkpoint from huggingface.co/meta-llama/Llama-3.2-1B-Instruct, licence required)
set -euo pipefail
GS=${1:-128}; EMB=${2:-4,32}; SEQ=${3:-384}; OUT=$(realpath -m "${4:-llama3_2_1b.pte}")
: "${ET_ROOT:?set ET_ROOT}"; : "${LLAMA_DIR:?set LLAMA_DIR}"
EMB_ARGS=(); [[ "$EMB" != "none" ]] && EMB_ARGS=(--embedding-quantize "$EMB")
cd "$ET_ROOT"
python -m examples.models.llama.export_llama \
  --model llama3_2 \
  --checkpoint "$LLAMA_DIR/consolidated.00.pth" --params "$LLAMA_DIR/params.json" \
  -kv --use_sdpa_with_kv_cache -X --xnnpack-extended-ops -d fp32 \
  -qmode 8da4w --group_size "$GS" "${EMB_ARGS[@]}" \
  --max_seq_length "$SEQ" --max_context_length "$SEQ" \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --output_name "$OUT"
echo "exported $OUT ($(du -h "$OUT" | cut -f1))"
