# =============================================================================
# DUSt3R × SemCom Makefile
# =============================================================================
#
# 常用指令速查
# ─────────────────────────────────────────────────────────────────────────────
# [Noise-only — 直接雜訊注入，無需訓練]
#  make exp              執行所有場景 × 所有通道的實驗
#  make exp-desktop      只跑 my_desktop（AWGN + Rayleigh）
#  make exp-boots        只跑 timberland_boots（AWGN + Rayleigh）
#
# [JSCC — DeepJSCC（凍結 backbone），需先訓練]
#  make train-jscc-awgn          訓練 AWGN JSCC（固定 SNR 10dB）
#  make train-jscc-rayleigh      訓練 Rayleigh JSCC（隨機 SNR 0~20dB）
#  make train-jscc-awgn-k256     壓縮比 1/4 版本（channel_dim=256）
#  make exp-jscc-desktop-awgn    JSCC 實驗：desktop × AWGN
#  make exp-jscc-desktop-rayleigh JSCC 實驗：desktop × Rayleigh
#  make exp-jscc-all             所有 JSCC 實驗
#
# [E2E — 端到端聯合訓練，需有深度資料集]
#  make train-e2e-awgn   E2E_DATASET="..."  訓練 AWGN
#  make train-e2e-rayleigh E2E_DATASET="..." 訓練 Rayleigh
#  make exp-e2e-desktop-awgn     E2E 實驗：desktop × AWGN
#  make exp-e2e-all              所有 E2E 實驗
#
# [繪圖]
#  make plot             畫出所有已完成實驗的圖
#  make plot-jscc        畫 JSCC vs noise-only 比較圖
#
# [Demo]
#  make demo             啟動原版 Gradio Demo
#  make demo-semcom      啟動 SemCom Demo（noise-only，port 7860）
#  make demo-jscc        啟動 SemCom Demo（JSCC checkpoint，port 7861）
#  make demo-e2e         啟動 SemCom Demo（E2E checkpoint，port 7862）
#
#  make clean            刪除所有結果 JSON 與圖片
#  make help             顯示這份說明
# =============================================================================

# ── 可調整的設定 ──────────────────────────────────────────────────────────────
WEIGHTS     := checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
GPU         := 1
IMAGE_SIZE  := 512
SNR_LIST    := inf 20 15 10 5 0
NITER       := 300
RESULTS_DIR := results
FIGURES_DIR := figures
PYTHON      := python

# JSCC / E2E 訓練超參數
CHANNEL_DIM      := 512          # 直接指定 channel symbol 數量
RATIO            ?=               # 或指定壓縮比（例如 0.5），會自動算出 channel_dim
                                  # 用法：make train-jscc-awgn RATIO=0.25
TRAIN_EPOCHS     := 200
TRAIN_LR         := 1e-3
TRAIN_SNR        := 10
TRAIN_SNR_RANGE  := 0 20
TRAIN_LOSS       := feat

# 壓縮比參數（根據是否設定 RATIO 自動選擇 --ratio 或 --channel_dim）
_DIM_ARG = $(if $(strip $(RATIO)),--ratio $(RATIO),--channel_dim $(CHANNEL_DIM))

# JSCC checkpoint 路徑
JSCC_AWGN_K512     := checkpoints/jscc_awgn_k$(CHANNEL_DIM).pth
JSCC_RAYLEIGH_K512 := checkpoints/jscc_rayleigh_k$(CHANNEL_DIM).pth
JSCC_AWGN_K256     := checkpoints/jscc_awgn_k256.pth

# JSCC 結果
DESKTOP_JSCC_AWGN_JSON     := $(RESULTS_DIR)/desktop_jscc_awgn.json
DESKTOP_JSCC_RAYLEIGH_JSON := $(RESULTS_DIR)/desktop_jscc_rayleigh.json
BOOTS_JSCC_AWGN_JSON       := $(RESULTS_DIR)/boots_jscc_awgn.json
BOOTS_JSCC_RAYLEIGH_JSON   := $(RESULTS_DIR)/boots_jscc_rayleigh.json

# E2E 訓練超參數（全部可訓練，無 freeze 選項）
E2E_EPOCHS         := 50
E2E_LR             := 1e-4
E2E_BACKBONE_SCALE := 0.1
E2E_BATCH          := 2
E2E_ACCUM          := 8
E2E_DATASET        ?=                   # 必須由使用者設定，例：BlendedMVS(...)

# E2E checkpoints（預設指向已訓練的模型；可在命令列覆蓋）
# 已訓練：e2e_awgn_snr0-20_r0.125 (k=128)  e2e_awgn_snr0-20_r0.083 (k=85)
E2E_AWGN     ?= checkpoints/e2e_awgn_snr0-20_r0.125/checkpoint-last.pth
E2E_RAYLEIGH ?= checkpoints/e2e_rayleigh_snr0-20_r0.125/checkpoint-last.pth

# E2E 結果
DESKTOP_E2E_AWGN_JSON     := $(RESULTS_DIR)/desktop_e2e_awgn.json
DESKTOP_E2E_RAYLEIGH_JSON := $(RESULTS_DIR)/desktop_e2e_rayleigh.json
BOOTS_E2E_AWGN_JSON       := $(RESULTS_DIR)/boots_e2e_awgn.json
BOOTS_E2E_RAYLEIGH_JSON   := $(RESULTS_DIR)/boots_e2e_rayleigh.json

# BlendedMVS 評估設定（用於 eval_semcom.py）
BMVS_ROOT    ?= data/blendedmvs_processed   # 必須由使用者設定
EVAL_SCRIPT  := $(PYTHON) eval_semcom.py
EVAL_SNR     := inf 20 15 10 5 0

# eval 結果路徑
EVAL_NOISY_AWGN_JSON    := $(RESULTS_DIR)/eval_noisy_awgn.json
EVAL_NOISY_RAYLEIGH_JSON:= $(RESULTS_DIR)/eval_noisy_rayleigh.json
EVAL_JSCC_AWGN_JSON     := $(RESULTS_DIR)/eval_jscc_awgn.json
EVAL_JSCC_RAYLEIGH_JSON := $(RESULTS_DIR)/eval_jscc_rayleigh.json
EVAL_E2E_AWGN_JSON      := $(RESULTS_DIR)/eval_e2e_awgn.json
EVAL_E2E_RAYLEIGH_JSON  := $(RESULTS_DIR)/eval_e2e_rayleigh.json

# ── 圖片目錄（用於 JSCC 訓練）
DESKTOP_DIR := dust3r/images/my_desktop
BOOTS_DIR   := dust3r/images/timberland_boots

# 圖片路徑（用於 Phase A 實驗）
DESKTOP_IMGS := dust3r/images/my_desktop/1.png \
                dust3r/images/my_desktop/2.png \
                dust3r/images/my_desktop/3.png \
                dust3r/images/my_desktop/4.png \
                dust3r/images/my_desktop/5.png

BOOTS_IMGS   := dust3r/images/timberland_boots/1.png \
                dust3r/images/timberland_boots/2.png \
                dust3r/images/timberland_boots/3.png \
                dust3r/images/timberland_boots/4.png \
                dust3r/images/timberland_boots/5.png

# ── 內部變數 ──────────────────────────────────────────────────────────────────
ENV          := CUDA_VISIBLE_DEVICES=$(GPU)
EXP_SCRIPT   := $(PYTHON) experiment_semcom.py
PLOT_SCRIPT  := $(PYTHON) plot_semcom.py
LOSS_SCRIPT  := $(PYTHON) plot_losses.py

# JSCC losses JSON paths（由 train_jscc.py 產生）
JSCC_AWGN_LOSSES     := $(basename $(JSCC_AWGN_K512))_losses.json
JSCC_RAYLEIGH_LOSSES := $(basename $(JSCC_RAYLEIGH_K512))_losses.json

DESKTOP_AWGN_JSON    := $(RESULTS_DIR)/desktop_awgn.json
DESKTOP_RAYLEIGH_JSON:= $(RESULTS_DIR)/desktop_rayleigh.json
BOOTS_AWGN_JSON      := $(RESULTS_DIR)/boots_awgn.json
BOOTS_RAYLEIGH_JSON  := $(RESULTS_DIR)/boots_rayleigh.json

.PHONY: all exp exp-desktop exp-boots \
        exp-desktop-awgn exp-desktop-rayleigh \
        exp-boots-awgn exp-boots-rayleigh \
        train-jscc-awgn train-jscc-rayleigh train-jscc-awgn-k256 \
        exp-jscc-desktop-awgn exp-jscc-desktop-rayleigh \
        exp-jscc-boots-awgn exp-jscc-boots-rayleigh exp-jscc-all \
        train-e2e-awgn train-e2e-rayleigh train-e2e-identity-awgn \
        eval-blendedmvs-clean eval-blendedmvs-identity-awgn \
        exp-e2e-desktop-awgn exp-e2e-desktop-rayleigh \
        exp-e2e-boots-awgn exp-e2e-boots-rayleigh exp-e2e-all \
        eval-blendedmvs-noisy-awgn eval-blendedmvs-noisy-rayleigh \
        eval-blendedmvs-jscc-awgn eval-blendedmvs-jscc-rayleigh \
        eval-blendedmvs-e2e-awgn eval-blendedmvs-e2e-rayleigh \
        eval-blendedmvs-awgn eval-blendedmvs-rayleigh eval-blendedmvs-all \
        plot-eval-awgn plot-eval-rayleigh \
        plot plot-desktop plot-boots plot-all-scenes \
        plot-jscc plot-jscc-desktop plot-jscc-boots plot-jscc-all-scenes \
        plot-e2e plot-e2e-desktop plot-e2e-boots plot-e2e-all-scenes \
        plot-losses plot-losses-awgn plot-losses-rayleigh \
        plot-compare-awgn plot-compare-rayleigh plot-compare \
        demo demo-semcom demo-jscc demo-e2e clean help dirs checkpoints

# ── 預設目標 ──────────────────────────────────────────────────────────────────
all: exp plot

help:
	@echo ""
	@echo "DUSt3R × SemCom — 可用指令"
	@echo "══════════════════════════════════════════════════════"
	@echo "【Noise-only — 直接雜訊注入，無需訓練】"
	@echo "  make exp                執行所有實驗（desktop + boots × awgn + rayleigh）"
	@echo "  make exp-desktop        只跑 my_desktop"
	@echo "  make exp-boots          只跑 timberland_boots"
	@echo "  make exp-desktop-awgn   只跑 my_desktop AWGN"
	@echo "  make exp-desktop-rayleigh 只跑 my_desktop Rayleigh"
	@echo ""
	@echo "【JSCC — DeepJSCC（凍結 backbone），需先訓練】"
	@echo "  make train-jscc-awgn              訓練 AWGN JSCC（k=$(CHANNEL_DIM), SNR=$(TRAIN_SNR)dB）"
	@echo "  make train-jscc-rayleigh          訓練 Rayleigh JSCC（k=$(CHANNEL_DIM), SNR=[$(TRAIN_SNR_RANGE)]dB）"
	@echo "  make train-jscc-awgn-k256         AWGN JSCC 壓縮比 1/4（k=256）"
	@echo "  make exp-jscc-desktop-awgn        JSCC 評估：desktop × AWGN"
	@echo "  make exp-jscc-desktop-rayleigh    JSCC 評估：desktop × Rayleigh"
	@echo "  make exp-jscc-boots-awgn          JSCC 評估：boots × AWGN"
	@echo "  make exp-jscc-boots-rayleigh      JSCC 評估：boots × Rayleigh"
	@echo "  make exp-jscc-all                 所有 JSCC 評估"
	@echo ""
	@echo "【E2E — 端到端聯合訓練，需有深度資料集】"
	@echo "  make train-e2e-awgn   E2E_DATASET=\"...\"   訓練 AWGN（需指定資料集）"
	@echo "  make train-e2e-rayleigh E2E_DATASET=\"...\" 訓練 Rayleigh"
	@echo "  make exp-e2e-desktop-awgn     E2E 定性評估：desktop × AWGN"
	@echo "  make exp-e2e-desktop-rayleigh E2E 定性評估：desktop × Rayleigh"
	@echo "  make exp-e2e-all              所有 E2E 定性評估"
	@echo ""
	@echo "【BlendedMVS 定量評估（task loss vs GT depth/pose）— 需設定 BMVS_ROOT】"
	@echo "  make eval-blendedmvs-all BMVS_ROOT=<path>  所有模型 × 所有通道（3路比較）"
	@echo "  make eval-blendedmvs-awgn BMVS_ROOT=<path> AWGN：noise-only / JSCC / E2E"
	@echo "  make eval-blendedmvs-rayleigh ...           Rayleigh：三路比較"
	@echo "  make plot-eval-awgn       繪製 AWGN 三路比較圖（task_loss + conf）"
	@echo "  make plot-eval-rayleigh   繪製 Rayleigh 三路比較圖"
	@echo ""
	@echo "【繪圖】"
	@echo "  make plot                Noise-only 全部圖（desktop / boots / awgn / rayleigh）"
	@echo "  make plot-jscc           JSCC 全部圖"
	@echo "  make plot-e2e            E2E 全部圖"
	@echo "  make plot-losses         訓練 loss 曲線（JSCC + E2E，各 channel 一張）"
	@echo "  make plot-losses-awgn    AWGN 訓練曲線（JSCC 200 epochs + E2E N epochs）"
	@echo "  make plot-losses-rayleigh Rayleigh 訓練曲線（JSCC）"
	@echo "  make plot-compare        三路比較：noise-only / JSCC / E2E（所有場景）"
	@echo "  make plot-compare-awgn   AWGN 三路比較（desktop + boots）"
	@echo "  make plot-compare-rayleigh Rayleigh 比較（desktop + boots）"
	@echo ""
	@echo "【Demo】"
	@echo "  make demo               啟動原版 Gradio Demo (port 7860)"
	@echo "  make demo-semcom        SemCom Demo，noise-only (port 7860, 0.0.0.0)"
	@echo "  make demo-jscc          SemCom Demo，JSCC checkpoint (port 7861, 0.0.0.0)"
	@echo "  make demo-e2e           SemCom Demo，E2E checkpoint (port 7862, 0.0.0.0)"
	@echo "  make clean              刪除所有結果與圖片"
	@echo ""
	@echo "  可調整參數（目前預設值）："
	@echo "    GPU=$(GPU)  IMAGE_SIZE=$(IMAGE_SIZE)  NITER=$(NITER)"
	@echo "    SNR_LIST=\"$(SNR_LIST)\""
	@echo "    CHANNEL_DIM=$(CHANNEL_DIM)  TRAIN_EPOCHS=$(TRAIN_EPOCHS)  TRAIN_LR=$(TRAIN_LR)"
	@echo "    TRAIN_LOSS=$(TRAIN_LOSS)  TRAIN_SNR=$(TRAIN_SNR)  TRAIN_SNR_RANGE=\"$(TRAIN_SNR_RANGE)\""
	@echo "    E2E_EPOCHS=$(E2E_EPOCHS)  E2E_LR=$(E2E_LR)"
	@echo "    E2E_BACKBONE_SCALE=$(E2E_BACKBONE_SCALE)  E2E_BATCH=$(E2E_BATCH)  E2E_ACCUM=$(E2E_ACCUM)"
	@echo "    E2E_DATASET=\"$(E2E_DATASET)\""
	@echo "══════════════════════════════════════════════════════"

# ── 建立輸出目錄 ──────────────────────────────────────────────────────────────
dirs:
	@mkdir -p $(RESULTS_DIR) $(FIGURES_DIR)

checkpoints:
	@mkdir -p checkpoints

# ── JSCC 訓練目標 ─────────────────────────────────────────────────────────────

# AWGN，固定 SNR=10dB，channel_dim=512（ratio=1/2），feature MSE loss
train-jscc-awgn: dirs checkpoints
	@echo "\n▶  訓練 JSCC：AWGN, SNR=$(TRAIN_SNR)dB, $(_DIM_ARG)"
	$(ENV) $(PYTHON) train_jscc.py \
		--weights      $(WEIGHTS) \
		--image_dirs   $(DESKTOP_DIR) $(BOOTS_DIR) \
		--channel      awgn \
		--snr_db       $(TRAIN_SNR) \
		$(_DIM_ARG) \
		--loss         $(TRAIN_LOSS) \
		--epochs       $(TRAIN_EPOCHS) \
		--lr           $(TRAIN_LR) \
		--image_size   $(IMAGE_SIZE) \
		--device       cuda \
		--output       $(JSCC_AWGN_K512)

# Rayleigh，隨機 SNR ∈ [0,20]dB，channel_dim=512，feature MSE loss
train-jscc-rayleigh: dirs checkpoints
	@echo "\n▶  訓練 JSCC：Rayleigh, SNR=[$(TRAIN_SNR_RANGE)]dB, $(_DIM_ARG)"
	$(ENV) $(PYTHON) train_jscc.py \
		--weights      $(WEIGHTS) \
		--image_dirs   $(DESKTOP_DIR) $(BOOTS_DIR) \
		--channel      rayleigh \
		--snr_range    $(TRAIN_SNR_RANGE) \
		$(_DIM_ARG) \
		--loss         $(TRAIN_LOSS) \
		--epochs       $(TRAIN_EPOCHS) \
		--lr           $(TRAIN_LR) \
		--image_size   $(IMAGE_SIZE) \
		--device       cuda \
		--output       $(JSCC_RAYLEIGH_K512)

# AWGN，channel_dim=256（ratio=1/4），feature MSE loss
train-jscc-awgn-k256: dirs checkpoints
	@echo "\n▶  訓練 JSCC：AWGN, SNR=$(TRAIN_SNR)dB, k=256 (ratio=1/4)"
	$(ENV) $(PYTHON) train_jscc.py \
		--weights      $(WEIGHTS) \
		--image_dirs   $(DESKTOP_DIR) $(BOOTS_DIR) \
		--channel      awgn \
		--snr_db       $(TRAIN_SNR) \
		--channel_dim  256 \
		--loss         $(TRAIN_LOSS) \
		--epochs       $(TRAIN_EPOCHS) \
		--lr           $(TRAIN_LR) \
		--image_size   $(IMAGE_SIZE) \
		--device       cuda \
		--output       $(JSCC_AWGN_K256)

# ── JSCC 實驗目標 ─────────────────────────────────────────────────────────────

exp-jscc-desktop-awgn: dirs $(JSCC_AWGN_K512)
	@echo "\n▶  JSCC 實驗：my_desktop × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(DESKTOP_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      awgn \
		--jscc_weights $(JSCC_AWGN_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(DESKTOP_JSCC_AWGN_JSON)

exp-jscc-desktop-rayleigh: dirs $(JSCC_RAYLEIGH_K512)
	@echo "\n▶  JSCC 實驗：my_desktop × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(DESKTOP_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      rayleigh \
		--jscc_weights $(JSCC_RAYLEIGH_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(DESKTOP_JSCC_RAYLEIGH_JSON)

exp-jscc-boots-awgn: dirs $(JSCC_AWGN_K512)
	@echo "\n▶  JSCC 實驗：timberland_boots × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(BOOTS_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      awgn \
		--jscc_weights $(JSCC_AWGN_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(BOOTS_JSCC_AWGN_JSON)

exp-jscc-boots-rayleigh: dirs $(JSCC_RAYLEIGH_K512)
	@echo "\n▶  JSCC 實驗：timberland_boots × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(BOOTS_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      rayleigh \
		--jscc_weights $(JSCC_RAYLEIGH_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(BOOTS_JSCC_RAYLEIGH_JSON)

exp-jscc-all: exp-jscc-desktop-awgn exp-jscc-desktop-rayleigh exp-jscc-boots-awgn exp-jscc-boots-rayleigh

# ── 實驗目標 ──────────────────────────────────────────────────────────────────
exp: exp-desktop exp-boots

exp-desktop: exp-desktop-awgn exp-desktop-rayleigh

exp-boots: exp-boots-awgn exp-boots-rayleigh

exp-desktop-awgn: dirs
	@echo "\n▶  my_desktop × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights   $(WEIGHTS) \
		--images    $(DESKTOP_IMGS) \
		--snr_list  $(SNR_LIST) \
		--channel   awgn \
		--image_size $(IMAGE_SIZE) \
		--niter     $(NITER) \
		--output    $(DESKTOP_AWGN_JSON)

exp-desktop-rayleigh: dirs
	@echo "\n▶  my_desktop × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights   $(WEIGHTS) \
		--images    $(DESKTOP_IMGS) \
		--snr_list  $(SNR_LIST) \
		--channel   rayleigh \
		--image_size $(IMAGE_SIZE) \
		--niter     $(NITER) \
		--output    $(DESKTOP_RAYLEIGH_JSON)

exp-boots-awgn: dirs
	@echo "\n▶  timberland_boots × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights   $(WEIGHTS) \
		--images    $(BOOTS_IMGS) \
		--snr_list  $(SNR_LIST) \
		--channel   awgn \
		--image_size $(IMAGE_SIZE) \
		--niter     $(NITER) \
		--output    $(BOOTS_AWGN_JSON)

exp-boots-rayleigh: dirs
	@echo "\n▶  timberland_boots × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights   $(WEIGHTS) \
		--images    $(BOOTS_IMGS) \
		--snr_list  $(SNR_LIST) \
		--channel   rayleigh \
		--image_size $(IMAGE_SIZE) \
		--niter     $(NITER) \
		--output    $(BOOTS_RAYLEIGH_JSON)

# ── 繪圖目標 ──────────────────────────────────────────────────────────────────
plot: plot-desktop plot-boots plot-all-scenes

plot-desktop: dirs
	@echo "\n▶  繪圖：my_desktop（Phase A）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_AWGN_JSON) $(DESKTOP_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/phaseA_desktop

plot-boots: dirs
	@echo "\n▶  繪圖：timberland_boots（Phase A）"
	$(PLOT_SCRIPT) \
		$(BOOTS_AWGN_JSON) $(BOOTS_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/phaseA_boots

plot-all-scenes: dirs
	@echo "\n▶  繪圖：兩場景 AWGN 比較（Phase A）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_AWGN_JSON) $(BOOTS_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/phaseA_awgn
	$(PLOT_SCRIPT) \
		$(DESKTOP_RAYLEIGH_JSON) $(BOOTS_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/phaseA_rayleigh

# JSCC 繪圖
plot-jscc: plot-jscc-desktop plot-jscc-boots plot-jscc-all-scenes

plot-jscc-desktop: dirs
	@echo "\n▶  繪圖：my_desktop（JSCC）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_JSCC_AWGN_JSON) $(DESKTOP_JSCC_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/jscc_desktop

plot-jscc-boots: dirs
	@echo "\n▶  繪圖：timberland_boots（JSCC）"
	$(PLOT_SCRIPT) \
		$(BOOTS_JSCC_AWGN_JSON) $(BOOTS_JSCC_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/jscc_boots

plot-jscc-all-scenes: dirs
	@echo "\n▶  繪圖：兩場景 AWGN 比較（JSCC）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_JSCC_AWGN_JSON) $(BOOTS_JSCC_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/jscc_awgn
	$(PLOT_SCRIPT) \
		$(DESKTOP_JSCC_RAYLEIGH_JSON) $(BOOTS_JSCC_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/jscc_rayleigh

# ── E2E 訓練目標 ──────────────────────────────────────────────────────────────

# 請先定義 E2E_DATASET：
#   make train-e2e-awgn E2E_DATASET="ScanNetpp(split='train', ROOT='data/scannetpp', resolution=512, aug_crop=16)"
train-e2e-awgn: dirs checkpoints
	@if [ -z "$(E2E_DATASET)" ]; then \
		echo "Error: E2E_DATASET 未設定！"; \
		echo "  用法: make train-e2e-awgn E2E_DATASET=\"BlendedMVS(split='train', ROOT='data/blendedmvs_processed', resolution=512, aug_crop=16)\""; \
		exit 1; \
	fi
	@echo "\n▶  訓練 E2E：AWGN, $(_DIM_ARG)"
	$(ENV) $(PYTHON) train_e2e.py \
		--weights            $(WEIGHTS) \
		--jscc_path          $(JSCC_AWGN_K512) \
		--dataset            "$(E2E_DATASET)" \
		--channel            awgn \
		--snr_range          $(TRAIN_SNR_RANGE) \
		$(_DIM_ARG) \
		--epochs             $(E2E_EPOCHS) \
		--lr                 $(E2E_LR) \
		--backbone_lr_scale  $(E2E_BACKBONE_SCALE) \
		--batch_size         $(E2E_BATCH) \
		--accum_iter         $(E2E_ACCUM) \
		--amp \
		--output_dir         checkpoints/e2e_awgn_k$(CHANNEL_DIM)/

# Rayleigh
train-e2e-rayleigh: dirs checkpoints
	@if [ -z "$(E2E_DATASET)" ]; then \
		echo "Error: E2E_DATASET 未設定！"; \
		echo "  用法: make train-e2e-rayleigh E2E_DATASET=\"BlendedMVS(split='train', ROOT='data/blendedmvs_processed', resolution=512, aug_crop=16)\""; \
		exit 1; \
	fi
	@echo "\n▶  訓練 E2E：Rayleigh, $(_DIM_ARG)"
	$(ENV) $(PYTHON) train_e2e.py \
		--weights            $(WEIGHTS) \
		--jscc_path          $(JSCC_RAYLEIGH_K512) \
		--dataset            "$(E2E_DATASET)" \
		--channel            rayleigh \
		--snr_range          $(TRAIN_SNR_RANGE) \
		$(_DIM_ARG) \
		--epochs             $(E2E_EPOCHS) \
		--lr                 $(E2E_LR) \
		--backbone_lr_scale  $(E2E_BACKBONE_SCALE) \
		--batch_size         $(E2E_BATCH) \
		--accum_iter         $(E2E_ACCUM) \
		--amp \
		--output_dir         checkpoints/e2e_rayleigh_k$(CHANNEL_DIM)/

# 乾淨 SemCom ablation baseline：fine-tune backbone + 注入雜訊，但「無 JSCC 壓縮」
# （identity channel）。與 E2E 配方相同，僅差壓縮，用來拆分 fine-tuning vs JSCC 貢獻。
train-e2e-identity-awgn: dirs checkpoints
	@if [ -z "$(E2E_DATASET)" ]; then \
		echo "Error: E2E_DATASET 未設定！"; \
		echo "  用法: make train-e2e-identity-awgn E2E_DATASET=\"10000 @ BlendedMVS(split='train', ROOT='data/blendedmvs_processed', resolution=512, aug_crop=16)\""; \
		exit 1; \
	fi
	@echo "\n▶  訓練 identity baseline（AWGN, 無 JSCC 壓縮）"
	$(ENV) $(PYTHON) train_e2e.py \
		--weights            $(WEIGHTS) \
		--dataset            "$(E2E_DATASET)" \
		--channel            awgn \
		--snr_range          $(TRAIN_SNR_RANGE) \
		--no_jscc \
		--epochs             $(E2E_EPOCHS) \
		--lr                 $(E2E_LR) \
		--backbone_lr_scale  $(E2E_BACKBONE_SCALE) \
		--batch_size         $(E2E_BATCH) \
		--accum_iter         $(E2E_ACCUM) \
		--amp \
		--output_dir         checkpoints/e2e_awgn_snr0-20_identity/

# ── E2E 評估目標 ──────────────────────────────────────────────────────────────

exp-e2e-desktop-awgn: dirs
	@echo "\n▶  E2E 評估：my_desktop × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_AWGN) \
		--images       $(DESKTOP_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      awgn \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(DESKTOP_E2E_AWGN_JSON)

exp-e2e-desktop-rayleigh: dirs
	@echo "\n▶  E2E 評估：my_desktop × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_RAYLEIGH) \
		--images       $(DESKTOP_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      rayleigh \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(DESKTOP_E2E_RAYLEIGH_JSON)

exp-e2e-boots-awgn: dirs
	@echo "\n▶  E2E 評估：timberland_boots × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_AWGN) \
		--images       $(BOOTS_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      awgn \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(BOOTS_E2E_AWGN_JSON)

exp-e2e-boots-rayleigh: dirs
	@echo "\n▶  E2E 評估：timberland_boots × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_RAYLEIGH) \
		--images       $(BOOTS_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      rayleigh \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(BOOTS_E2E_RAYLEIGH_JSON)

exp-e2e-all: exp-e2e-desktop-awgn exp-e2e-desktop-rayleigh \
             exp-e2e-boots-awgn   exp-e2e-boots-rayleigh

# ── BlendedMVS 資料集評估目標 ─────────────────────────────────────────────────
#
# 使用 GT depth + camera pose 計算 task loss（ConfLoss + Regr3D），
# 比 experiment_semcom.py 更嚴格。
#
# 需先設定 BMVS_ROOT：
#   make eval-blendedmvs-all BMVS_ROOT=data/blendedmvs_processed

_BMVS_DATASET = "BlendedMVS(split='val', ROOT='$(BMVS_ROOT)', resolution=$(IMAGE_SIZE), aug_crop=16)"

eval-blendedmvs-noisy-awgn: dirs
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 noise-only：BlendedMVS val × AWGN"
	$(ENV) $(EVAL_SCRIPT) \
		--weights  $(WEIGHTS) \
		--channel  awgn \
		--snr_list $(EVAL_SNR) \
		--dataset  $(_BMVS_DATASET) \
		--output   $(EVAL_NOISY_AWGN_JSON)

eval-blendedmvs-noisy-rayleigh: dirs
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 noise-only：BlendedMVS val × Rayleigh"
	$(ENV) $(EVAL_SCRIPT) \
		--weights  $(WEIGHTS) \
		--channel  rayleigh \
		--snr_list $(EVAL_SNR) \
		--dataset  $(_BMVS_DATASET) \
		--output   $(EVAL_NOISY_RAYLEIGH_JSON)

eval-blendedmvs-jscc-awgn: dirs $(JSCC_AWGN_K512)
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 JSCC-only：BlendedMVS val × AWGN"
	$(ENV) $(EVAL_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(JSCC_AWGN_K512) \
		--channel      awgn \
		--snr_list     $(EVAL_SNR) \
		--dataset      $(_BMVS_DATASET) \
		--output       $(EVAL_JSCC_AWGN_JSON)

eval-blendedmvs-jscc-rayleigh: dirs $(JSCC_RAYLEIGH_K512)
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 JSCC-only：BlendedMVS val × Rayleigh"
	$(ENV) $(EVAL_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(JSCC_RAYLEIGH_K512) \
		--channel      rayleigh \
		--snr_list     $(EVAL_SNR) \
		--dataset      $(_BMVS_DATASET) \
		--output       $(EVAL_JSCC_RAYLEIGH_JSON)

eval-blendedmvs-e2e-awgn: dirs
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 E2E：BlendedMVS val × AWGN"
	$(ENV) $(EVAL_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_AWGN) \
		--channel      awgn \
		--snr_list     $(EVAL_SNR) \
		--dataset      $(_BMVS_DATASET) \
		--output       $(EVAL_E2E_AWGN_JSON)

eval-blendedmvs-e2e-rayleigh: dirs
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 E2E：BlendedMVS val × Rayleigh"
	$(ENV) $(EVAL_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_RAYLEIGH) \
		--channel      rayleigh \
		--snr_list     $(EVAL_SNR) \
		--dataset      $(_BMVS_DATASET) \
		--output       $(EVAL_E2E_RAYLEIGH_JSON)

# Clean DUSt3R upper-bound（無 channel、無壓縮，只跑 inf SNR）→ 繪圖時當參考線
eval-blendedmvs-clean: dirs
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 clean DUSt3R upper-bound：BlendedMVS val（inf only）"
	$(ENV) $(EVAL_SCRIPT) \
		--weights  $(WEIGHTS) \
		--channel  awgn \
		--snr_list inf \
		--dataset  $(_BMVS_DATASET) \
		--output   $(RESULTS_DIR)/eval_clean_upperbound.json

# 乾淨 SemCom baseline（identity，無 JSCC）的評估
eval-blendedmvs-identity-awgn: dirs
	@if [ -z "$(BMVS_ROOT)" ] || [ "$(BMVS_ROOT)" = "data/blendedmvs_processed" ]; then \
		echo "請設定 BMVS_ROOT：make $@ BMVS_ROOT=<path>"; exit 1; fi
	@echo "\n▶  評估 identity baseline：BlendedMVS val × AWGN"
	$(ENV) $(EVAL_SCRIPT) \
		--weights      $(WEIGHTS) \
		--jscc_weights checkpoints/e2e_awgn_snr0-20_identity/checkpoint-last.pth \
		--channel      awgn \
		--snr_list     $(EVAL_SNR) \
		--dataset      $(_BMVS_DATASET) \
		--output       $(RESULTS_DIR)/eval_identity_awgn.json

eval-blendedmvs-awgn: eval-blendedmvs-noisy-awgn eval-blendedmvs-jscc-awgn eval-blendedmvs-e2e-awgn
eval-blendedmvs-rayleigh: eval-blendedmvs-noisy-rayleigh eval-blendedmvs-jscc-rayleigh eval-blendedmvs-e2e-rayleigh
eval-blendedmvs-all: eval-blendedmvs-awgn eval-blendedmvs-rayleigh

# 繪製 eval 結果（三路比較：noise-only / JSCC / E2E）
plot-eval-awgn: dirs
	@echo "\n▶  繪圖：BlendedMVS val AWGN 評估比較"
	$(PLOT_SCRIPT) \
		$(EVAL_NOISY_AWGN_JSON) \
		$(EVAL_JSCC_AWGN_JSON) \
		$(EVAL_E2E_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/eval_awgn

plot-eval-rayleigh: dirs
	@echo "\n▶  繪圖：BlendedMVS val Rayleigh 評估比較"
	$(PLOT_SCRIPT) \
		$(EVAL_NOISY_RAYLEIGH_JSON) \
		$(EVAL_JSCC_RAYLEIGH_JSON) \
		$(EVAL_E2E_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/eval_rayleigh

# ── E2E 繪圖目標 ──────────────────────────────────────────────────────────────

plot-e2e: plot-e2e-desktop plot-e2e-boots plot-e2e-all-scenes

plot-e2e-desktop: dirs
	@echo "\n▶  繪圖：my_desktop（E2E）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_E2E_AWGN_JSON) $(DESKTOP_E2E_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/e2e_desktop

plot-e2e-boots: dirs
	@echo "\n▶  繪圖：timberland_boots（E2E）"
	$(PLOT_SCRIPT) \
		$(BOOTS_E2E_AWGN_JSON) $(BOOTS_E2E_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/e2e_boots

plot-e2e-all-scenes: dirs
	@echo "\n▶  繪圖：兩場景 AWGN 比較（E2E）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_E2E_AWGN_JSON) $(BOOTS_E2E_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/e2e_awgn
	$(PLOT_SCRIPT) \
		$(DESKTOP_E2E_RAYLEIGH_JSON) $(BOOTS_E2E_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/e2e_rayleigh

# ── 訓練 Loss 曲線 ─────────────────────────────────────────────────────────────

# AWGN：JSCC（200 epochs）+ E2E（N epochs）放在同一張圖
plot-losses-awgn: dirs
	@echo "\n▶  繪製訓練 Loss 曲線（AWGN）"
	$(LOSS_SCRIPT) \
		$(JSCC_AWGN_LOSSES) \
		$(E2E_AWGN) \
		--outdir $(FIGURES_DIR)/losses_awgn

# Rayleigh：只有 JSCC 有 losses JSON
plot-losses-rayleigh: dirs
	@echo "\n▶  繪製訓練 Loss 曲線（Rayleigh）"
	$(LOSS_SCRIPT) \
		$(JSCC_RAYLEIGH_LOSSES) \
		--outdir $(FIGURES_DIR)/losses_rayleigh

plot-losses: plot-losses-awgn plot-losses-rayleigh

# ── 三路比較圖（noise-only / JSCC / E2E） ─────────────────────────────────────

# AWGN：三路比較，desktop 場景
plot-compare-desktop-awgn: dirs
	@echo "\n▶  三路比較：my_desktop × AWGN（noise-only + JSCC + E2E）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_AWGN_JSON) \
		$(DESKTOP_JSCC_AWGN_JSON) \
		$(DESKTOP_E2E_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/compare_desktop_awgn

# Rayleigh：三路比較，desktop 場景
plot-compare-desktop-rayleigh: dirs
	@echo "\n▶  三路比較：my_desktop × Rayleigh（noise-only + JSCC）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_RAYLEIGH_JSON) \
		$(DESKTOP_JSCC_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/compare_desktop_rayleigh

# AWGN：三路比較，boots 場景
plot-compare-boots-awgn: dirs
	@echo "\n▶  三路比較：timberland_boots × AWGN（noise-only + JSCC + E2E）"
	$(PLOT_SCRIPT) \
		$(BOOTS_AWGN_JSON) \
		$(BOOTS_JSCC_AWGN_JSON) \
		$(BOOTS_E2E_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/compare_boots_awgn

# Rayleigh：三路比較，boots 場景
plot-compare-boots-rayleigh: dirs
	@echo "\n▶  三路比較：timberland_boots × Rayleigh（noise-only + JSCC）"
	$(PLOT_SCRIPT) \
		$(BOOTS_RAYLEIGH_JSON) \
		$(BOOTS_JSCC_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/compare_boots_rayleigh

plot-compare-awgn: plot-compare-desktop-awgn plot-compare-boots-awgn
plot-compare-rayleigh: plot-compare-desktop-rayleigh plot-compare-boots-rayleigh
plot-compare: plot-compare-awgn plot-compare-rayleigh

# ── Demo ─────────────────────────────────────────────────────────────────────
demo:
	@echo "\n▶  啟動 Gradio Demo（GPU=$(GPU)，port 7860）"
	$(ENV) $(PYTHON) demo.py \
		--weights    $(WEIGHTS) \
		--image_size $(IMAGE_SIZE)

# noise-only baseline（不需要 JSCC checkpoint）
demo-semcom:
	@echo "\n▶  啟動 SemCom Demo（noise-only，GPU=$(GPU)，port 7860）"
	$(ENV) $(PYTHON) demo_semcom.py \
		--weights      $(WEIGHTS) \
		--image_size   $(IMAGE_SIZE) \
		--server_port  7860 \
		--local_network

# JSCC（凍結 backbone）
demo-jscc:
	@echo "\n▶  啟動 SemCom Demo（JSCC，GPU=$(GPU)，port 7861）"
	$(ENV) $(PYTHON) demo_semcom.py \
		--weights      $(WEIGHTS) \
		--jscc_weights $(JSCC_AWGN_K512) \
		--image_size   $(IMAGE_SIZE) \
		--server_port  7861 \
		--local_network

# E2E（端到端聯合訓練）
demo-e2e:
	@echo "\n▶  啟動 SemCom Demo（E2E，GPU=$(GPU)，port 7862）"
	$(ENV) $(PYTHON) demo_semcom.py \
		--weights      $(WEIGHTS) \
		--jscc_weights $(E2E_AWGN) \
		--image_size   $(IMAGE_SIZE) \
		--server_port  7862 \
		--local_network

# ── 清理 ─────────────────────────────────────────────────────────────────────
clean:
	@echo "▶  刪除 $(RESULTS_DIR)/ 與 $(FIGURES_DIR)/"
	rm -rf $(RESULTS_DIR) $(FIGURES_DIR)
