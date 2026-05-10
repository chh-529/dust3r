# =============================================================================
# DUSt3R × SemCom — Phase A Experiment Makefile
# =============================================================================
#
# 常用指令速查
# ─────────────────────────────────────────────────────────────────────────────
#  make exp              執行所有場景 × 所有通道的實驗
#  make exp-desktop      只跑 my_desktop（AWGN + Rayleigh）
#  make exp-boots        只跑 timberland_boots（AWGN + Rayleigh）
#
#  make plot             畫出所有已完成實驗的圖
#  make plot-desktop     只畫 my_desktop 的結果
#  make plot-boots       只畫 timberland_boots 的結果
#  make plot-compare     把 AWGN vs Rayleigh 畫在同一張圖上（各場景）
#  make plot-all-scenes  把兩個場景的 AWGN 結果畫在同一張圖上
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

# 圖片路徑
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
        plot plot-desktop plot-boots plot-compare plot-all-scenes \
        demo clean help dirs

# ── 預設目標 ──────────────────────────────────────────────────────────────────
all: exp plot

help:
	@echo ""
	@echo "DUSt3R × SemCom Phase A — 可用指令"
	@echo "══════════════════════════════════════════════════════"
	@echo "  make exp                執行所有實驗（desktop + boots × awgn + rayleigh）"
	@echo "  make exp-desktop        只跑 my_desktop"
	@echo "  make exp-boots          只跑 timberland_boots"
	@echo "  make exp-desktop-awgn   只跑 my_desktop AWGN"
	@echo "  make exp-desktop-rayleigh 只跑 my_desktop Rayleigh"
	@echo "  make exp-boots-awgn     只跑 timberland_boots AWGN"
	@echo "  make exp-boots-rayleigh 只跑 timberland_boots Rayleigh"
	@echo ""
	@echo "  make plot               畫出所有結果"
	@echo "  make plot-desktop       畫 my_desktop 結果"
	@echo "  make plot-boots         畫 timberland_boots 結果"
	@echo "  make plot-compare       AWGN vs Rayleigh 比較圖（各場景）"
	@echo "  make plot-all-scenes    兩場景 AWGN 結果比較"
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
plot: plot-desktop plot-boots plot-compare plot-all-scenes

plot-desktop: dirs
	@echo "\n▶  繪圖：my_desktop"
	$(PLOT_SCRIPT) \
		$(DESKTOP_AWGN_JSON) $(DESKTOP_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/desktop

plot-boots: dirs
	@echo "\n▶  繪圖：timberland_boots"
	$(PLOT_SCRIPT) \
		$(BOOTS_AWGN_JSON) $(BOOTS_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/boots

plot-compare: dirs
	@echo "\n▶  繪圖：AWGN vs Rayleigh（各場景合併）"
	$(PLOT_SCRIPT) \
		$(DESKTOP_AWGN_JSON) $(DESKTOP_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/compare_desktop
	$(PLOT_SCRIPT) \
		$(BOOTS_AWGN_JSON) $(BOOTS_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/compare_boots

plot-all-scenes: dirs
	@echo "\n▶  繪圖：兩場景 AWGN 比較"
	$(PLOT_SCRIPT) \
		$(DESKTOP_AWGN_JSON) $(BOOTS_AWGN_JSON) \
		--outdir $(FIGURES_DIR)/all_scenes_awgn
	$(PLOT_SCRIPT) \
		$(DESKTOP_RAYLEIGH_JSON) $(BOOTS_RAYLEIGH_JSON) \
		--outdir $(FIGURES_DIR)/all_scenes_rayleigh

# ── Demo ─────────────────────────────────────────────────────────────────────
demo:
	@echo "\n▶  啟動 Gradio Demo（GPU=$(GPU)，port 7860）"
	$(ENV) $(PYTHON) demo.py \
		--weights    $(WEIGHTS) \
		--image_size $(IMAGE_SIZE)

# ── 清理 ─────────────────────────────────────────────────────────────────────
clean:
	@echo "▶  刪除 $(RESULTS_DIR)/ 與 $(FIGURES_DIR)/"
	rm -rf $(RESULTS_DIR) $(FIGURES_DIR)
