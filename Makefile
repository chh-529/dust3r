# =============================================================================
# DUSt3R × SemCom — Phase A & B Experiment Makefile
# =============================================================================
#
# 常用指令速查
# ─────────────────────────────────────────────────────────────────────────────
# [Phase A — 直接雜訊注入，無需訓練]
#  make exp              執行所有場景 × 所有通道的實驗
#  make exp-desktop      只跑 my_desktop（AWGN + Rayleigh）
#  make exp-boots        只跑 timberland_boots（AWGN + Rayleigh）
#
# [Phase B — Linear JSCC，需先訓練]
#  make train-phaseB-awgn          訓練 AWGN JSCC（固定 SNR 10dB）
#  make train-phaseB-rayleigh      訓練 Rayleigh JSCC（隨機 SNR 0~20dB）
#  make train-phaseB-awgn-k256     壓縮比 1/4 版本（channel_dim=256）
#  make exp-phaseB-desktop-awgn    Phase B 實驗：desktop × AWGN
#  make exp-phaseB-desktop-rayleigh Phase B 實驗：desktop × Rayleigh
#  make exp-phaseB-all             所有 Phase B 實驗
#
# [繪圖]
#  make plot             畫出所有已完成實驗的圖
#  make plot-phaseB      畫 Phase B vs Phase A 比較圖
#
#  make demo             啟動 Gradio 互動 Demo
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

# Phase B JSCC 訓練超參數
CHANNEL_DIM      := 512
TRAIN_EPOCHS     := 200
TRAIN_LR         := 1e-3
TRAIN_SNR        := 10
TRAIN_SNR_RANGE  := 0 20
TRAIN_LOSS       := feat

# JSCC checkpoint 路徑
JSCC_AWGN_K512     := checkpoints/jscc_phaseB_awgn_k$(CHANNEL_DIM).pth
JSCC_RAYLEIGH_K512 := checkpoints/jscc_phaseB_rayleigh_k$(CHANNEL_DIM).pth
JSCC_AWGN_K256     := checkpoints/jscc_phaseB_awgn_k256.pth

# Phase B 結果
DESKTOP_PHASEB_AWGN_JSON     := $(RESULTS_DIR)/desktop_phaseB_awgn.json
DESKTOP_PHASEB_RAYLEIGH_JSON := $(RESULTS_DIR)/desktop_phaseB_rayleigh.json
BOOTS_PHASEB_AWGN_JSON       := $(RESULTS_DIR)/boots_phaseB_awgn.json
BOOTS_PHASEB_RAYLEIGH_JSON   := $(RESULTS_DIR)/boots_phaseB_rayleigh.json

# 圖片目錄（用於 Phase B 訓練）
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

DESKTOP_AWGN_JSON    := $(RESULTS_DIR)/desktop_awgn.json
DESKTOP_RAYLEIGH_JSON:= $(RESULTS_DIR)/desktop_rayleigh.json
BOOTS_AWGN_JSON      := $(RESULTS_DIR)/boots_awgn.json
BOOTS_RAYLEIGH_JSON  := $(RESULTS_DIR)/boots_rayleigh.json

.PHONY: all exp exp-desktop exp-boots \
        exp-desktop-awgn exp-desktop-rayleigh \
        exp-boots-awgn exp-boots-rayleigh \
        train-phaseB-awgn train-phaseB-rayleigh train-phaseB-awgn-k256 \
        exp-phaseB-desktop-awgn exp-phaseB-desktop-rayleigh \
        exp-phaseB-boots-awgn exp-phaseB-boots-rayleigh exp-phaseB-all \
        plot plot-desktop plot-boots plot-all-scenes \
        plot-phaseB plot-phaseB-desktop plot-phaseB-boots plot-phaseB-all-scenes \
        demo demo-semcom demo-semcom-phaseB clean help dirs checkpoints

# ── 預設目標 ──────────────────────────────────────────────────────────────────
all: exp plot

help:
	@echo ""
	@echo "DUSt3R × SemCom — 可用指令"
	@echo "══════════════════════════════════════════════════════"
	@echo "【Phase A — 直接雜訊注入，無需訓練】"
	@echo "  make exp                執行所有實驗（desktop + boots × awgn + rayleigh）"
	@echo "  make exp-desktop        只跑 my_desktop"
	@echo "  make exp-boots          只跑 timberland_boots"
	@echo "  make exp-desktop-awgn   只跑 my_desktop AWGN"
	@echo "  make exp-desktop-rayleigh 只跑 my_desktop Rayleigh"
	@echo ""
	@echo "【Phase B — DeepJSCC，需先訓練】"
	@echo "  make train-phaseB-awgn              訓練 AWGN JSCC（k=$(CHANNEL_DIM), SNR=$(TRAIN_SNR)dB）"
	@echo "  make train-phaseB-rayleigh          訓練 Rayleigh JSCC（k=$(CHANNEL_DIM), SNR=[$(TRAIN_SNR_RANGE)]dB）"
	@echo "  make train-phaseB-awgn-k256         AWGN JSCC 壓縮比 1/4（k=256）"
	@echo "  make exp-phaseB-desktop-awgn        Phase B 評估：desktop × AWGN"
	@echo "  make exp-phaseB-desktop-rayleigh    Phase B 評估：desktop × Rayleigh"
	@echo "  make exp-phaseB-boots-awgn          Phase B 評估：boots × AWGN"
	@echo "  make exp-phaseB-boots-rayleigh      Phase B 評估：boots × Rayleigh"
	@echo "  make exp-phaseB-all                 所有 Phase B 評估"
	@echo ""
	@echo "【繪圖】"
	@echo "  make plot                   Phase A 全部圖（desktop / boots / awgn / rayleigh）"
	@echo "  make plot-desktop           phaseA_desktop/（desktop AWGN vs Rayleigh）"
	@echo "  make plot-boots             phaseA_boots/（boots AWGN vs Rayleigh）"
	@echo "  make plot-all-scenes        phaseA_awgn/ 與 phaseA_rayleigh/（兩場景比較）"
	@echo "  make plot-phaseB            Phase B 全部圖（與 Phase A 對稱結構）"
	@echo "  make plot-phaseB-desktop    phaseB_desktop/（desktop AWGN vs Rayleigh）"
	@echo "  make plot-phaseB-boots      phaseB_boots/（boots AWGN vs Rayleigh）"
	@echo "  make plot-phaseB-all-scenes phaseB_awgn/ 與 phaseB_rayleigh/（兩場景比較）"
	@echo ""
	@echo "  make demo               啟動 Gradio Demo (port 7860)"
	@echo "  make clean              刪除所有結果與圖片"
	@echo ""
	@echo "  可調整參數（目前預設值）："
	@echo "    GPU=$(GPU)  IMAGE_SIZE=$(IMAGE_SIZE)  NITER=$(NITER)"
	@echo "    SNR_LIST=\"$(SNR_LIST)\""
	@echo "    CHANNEL_DIM=$(CHANNEL_DIM)  TRAIN_EPOCHS=$(TRAIN_EPOCHS)  TRAIN_LR=$(TRAIN_LR)"
	@echo "    TRAIN_LOSS=$(TRAIN_LOSS)  TRAIN_SNR=$(TRAIN_SNR)  TRAIN_SNR_RANGE=\"$(TRAIN_SNR_RANGE)\""
	@echo "══════════════════════════════════════════════════════"
	@echo ""
	@echo "  make demo               啟動 Gradio Demo (port 7860)"
	@echo "  make clean              刪除所有結果與圖片"
	@echo ""
	@echo "  可調整參數（預設值）："
	@echo "    GPU=$(GPU)  IMAGE_SIZE=$(IMAGE_SIZE)  NITER=$(NITER)"
	@echo "    SNR_LIST=\"$(SNR_LIST)\""
	@echo "══════════════════════════════════════════════════════"

# ── 建立輸出目錄 ──────────────────────────────────────────────────────────────
dirs:
	@mkdir -p $(RESULTS_DIR) $(FIGURES_DIR)

checkpoints:
	@mkdir -p checkpoints

# ── Phase B 訓練目標 ───────────────────────────────────────────────────────────

# AWGN，固定 SNR=10dB，channel_dim=512（ratio=1/2），feature MSE loss
train-phaseB-awgn: dirs checkpoints
	@echo "\n▶  訓練 Phase B JSCC：AWGN, SNR=$(TRAIN_SNR)dB, k=$(CHANNEL_DIM)"
	$(ENV) $(PYTHON) train_semcom_phaseB.py \
		--weights      $(WEIGHTS) \
		--image_dirs   $(DESKTOP_DIR) $(BOOTS_DIR) \
		--channel      awgn \
		--snr_db       $(TRAIN_SNR) \
		--channel_dim  $(CHANNEL_DIM) \
		--loss         $(TRAIN_LOSS) \
		--epochs       $(TRAIN_EPOCHS) \
		--lr           $(TRAIN_LR) \
		--image_size   $(IMAGE_SIZE) \
		--device       cuda \
		--output       $(JSCC_AWGN_K512)

# Rayleigh，隨機 SNR ∈ [0,20]dB，channel_dim=512，feature MSE loss
train-phaseB-rayleigh: dirs checkpoints
	@echo "\n▶  訓練 Phase B JSCC：Rayleigh, SNR=[$(TRAIN_SNR_RANGE)]dB, k=$(CHANNEL_DIM)"
	$(ENV) $(PYTHON) train_semcom_phaseB.py \
		--weights      $(WEIGHTS) \
		--image_dirs   $(DESKTOP_DIR) $(BOOTS_DIR) \
		--channel      rayleigh \
		--snr_range    $(TRAIN_SNR_RANGE) \
		--channel_dim  $(CHANNEL_DIM) \
		--loss         $(TRAIN_LOSS) \
		--epochs       $(TRAIN_EPOCHS) \
		--lr           $(TRAIN_LR) \
		--image_size   $(IMAGE_SIZE) \
		--device       cuda \
		--output       $(JSCC_RAYLEIGH_K512)

# AWGN，channel_dim=256（ratio=1/4），feature MSE loss
train-phaseB-awgn-k256: dirs checkpoints
	@echo "\n▶  訓練 Phase B JSCC：AWGN, SNR=$(TRAIN_SNR)dB, k=256 (ratio=1/4)"
	$(ENV) $(PYTHON) train_semcom_phaseB.py \
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

# ── Phase B 實驗目標 ───────────────────────────────────────────────────────────

exp-phaseB-desktop-awgn: dirs $(JSCC_AWGN_K512)
	@echo "\n▶  Phase B 實驗：my_desktop × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(DESKTOP_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      awgn \
		--jscc_weights $(JSCC_AWGN_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(DESKTOP_PHASEB_AWGN_JSON)

exp-phaseB-desktop-rayleigh: dirs $(JSCC_RAYLEIGH_K512)
	@echo "\n▶  Phase B 實驗：my_desktop × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(DESKTOP_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      rayleigh \
		--jscc_weights $(JSCC_RAYLEIGH_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(DESKTOP_PHASEB_RAYLEIGH_JSON)

exp-phaseB-boots-awgn: dirs $(JSCC_AWGN_K512)
	@echo "\n▶  Phase B 實驗：timberland_boots × AWGN"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(BOOTS_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      awgn \
		--jscc_weights $(JSCC_AWGN_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(BOOTS_PHASEB_AWGN_JSON)

exp-phaseB-boots-rayleigh: dirs $(JSCC_RAYLEIGH_K512)
	@echo "\n▶  Phase B 實驗：timberland_boots × Rayleigh"
	$(ENV) $(EXP_SCRIPT) \
		--weights      $(WEIGHTS) \
		--images       $(BOOTS_IMGS) \
		--snr_list     $(SNR_LIST) \
		--channel      rayleigh \
		--jscc_weights $(JSCC_RAYLEIGH_K512) \
		--image_size   $(IMAGE_SIZE) \
		--niter        $(NITER) \
		--output       $(BOOTS_PHASEB_RAYLEIGH_JSON)

exp-phaseB-all: exp-phaseB-desktop-awgn exp-phaseB-desktop-rayleigh exp-phaseB-boots-awgn exp-phaseB-boots-rayleigh

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

# Phase B 繪圖（與 Phase A 對稱的結構）
plot-phaseB: plot-phaseB-desktop plot-phaseB-boots plot-phaseB-all-scenes

plot-phaseB-desktop: dirs
	@echo "\n▶  繪圖：my_desktop（Phase B）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_PHASEB_AWGN_JSON) $(DESKTOP_PHASEB_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/phaseB_desktop

plot-phaseB-boots: dirs
	@echo "\n▶  繪圖：timberland_boots（Phase B）"
	$(PLOT_SCRIPT) \
		$(BOOTS_PHASEB_AWGN_JSON) $(BOOTS_PHASEB_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/phaseB_boots

plot-phaseB-all-scenes: dirs
	@echo "\n▶  繪圖：兩場景 AWGN 比較（Phase B）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_PHASEB_AWGN_JSON) $(BOOTS_PHASEB_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/phaseB_awgn
	$(PLOT_SCRIPT) \
		$(DESKTOP_PHASEB_RAYLEIGH_JSON) $(BOOTS_PHASEB_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/phaseB_rayleigh

# ── Demo ─────────────────────────────────────────────────────────────────────
demo:
	@echo "\n▶  啟動 Gradio Demo（GPU=$(GPU)，port 7860）"
	$(ENV) $(PYTHON) demo.py \
		--weights    $(WEIGHTS) \
		--image_size $(IMAGE_SIZE)

demo-semcom:
	@echo "\n▶  啟動 SemCom Demo（GPU=$(GPU)，port 7861，Phase A）"
	$(ENV) $(PYTHON) demo_semcom.py \
		--weights    $(WEIGHTS) \
		--image_size $(IMAGE_SIZE) \
		--server_port 7860

demo-semcom-phaseB:
	@echo "\n▶  啟動 SemCom Demo（GPU=$(GPU)，port 7861，Phase B）"
	$(ENV) $(PYTHON) demo_semcom.py \
		--weights      $(WEIGHTS) \
		--jscc_weights $(JSCC_AWGN_K512) \
		--image_size   $(IMAGE_SIZE) \
		--server_port  7861

# ── 清理 ─────────────────────────────────────────────────────────────────────
clean:
	@echo "▶  刪除 $(RESULTS_DIR)/ 與 $(FIGURES_DIR)/"
	rm -rf $(RESULTS_DIR) $(FIGURES_DIR)
