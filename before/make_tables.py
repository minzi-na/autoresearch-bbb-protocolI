import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import numpy as np

# Korean font
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Noto Sans CJK KR'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# TABLE 1: 점수 개선 추이 (keep된 실험)
# ─────────────────────────────────────────────
def make_table1():
    headers = ['#', 'commit', 'score_rs', 'score_s', 'avg', '변경 내용']
    rows = [
        ['1', 'e6d79c1', '2.5410', '2.3705', '2.456', '베이스라인: 8-feat gMLP, gated pool'],
        ['2', 'd1e1e0e', '2.5661', '2.3414', '2.454', 'modal_drop_p + cond_type + stability 수정, 3-seed 재베이스라인'],
        ['3', '6818e87', '2.5671', '2.3414', '2.454', '블록별 학습 가능한 res_scale 추가 (미미한 개선)'],
        ['4', '029f163', '2.6063', '2.3634', '2.485', 'label smoothing ls_eps [0, 0.15] — rs +0.039, s +0.022'],
        ['5', '5c2c7cb', '2.5991', '2.4065', '2.503', 'ls_eps [0, 0.25] + modal_drop [0, 0.4] 서치 공간 확장 — s +0.043'],
        ['6', '28d0009', '2.6185', '2.3966', '2.508', 'projection 후 LN-only — avg 2.507 신기록'],
        ['7', '1a3b37d', '2.6179', '2.4255', '2.522', 'composite-score early stopping — s +0.028, avg 신기록'],
    ]

    col_widths = [0.03, 0.07, 0.08, 0.07, 0.06, 0.55]
    fig_w = 18
    row_h = 0.60
    header_h = 0.70
    top_pad = 0.9
    bottom_pad = 0.4

    n_rows = len(rows)
    fig_h = top_pad + header_h + n_rows * row_h + bottom_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    title_y = fig_h - 0.35
    ax.text(0.5, title_y, '점수 개선 추이 (keep된 실험)',
            ha='center', va='center', fontsize=18, fontweight='bold',
            fontproperties=font_prop)

    # column x positions
    xs = []
    x = 0.01
    for w in col_widths:
        xs.append(x)
        x += w
    xs_center = [xs[i] + col_widths[i]/2 for i in range(len(col_widths))]

    # header row
    hy = fig_h - top_pad - header_h/2
    for i, (h, xc) in enumerate(zip(headers, xs_center)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs[i]+0.002, hy - header_h/2 + 0.04),
            col_widths[i] - 0.004, header_h - 0.05,
            boxstyle="round,pad=0.01", linewidth=0,
            facecolor='#2C3E50', zorder=2))
        ax.text(xc, hy, h, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white',
                fontproperties=font_prop, zorder=3)

    # data rows
    highlight_avgs = {'2.485', '2.503', '2.508', '2.522'}
    for r_idx, row in enumerate(rows):
        y = fig_h - top_pad - header_h - (r_idx + 0.5) * row_h
        bg = '#ECF0F1' if r_idx % 2 == 0 else 'white'
        # highlight best rows
        if row[4] in highlight_avgs:
            bg = '#D5F5E3'
        ax.add_patch(mpatches.Rectangle(
            (0.005, y - row_h/2 + 0.03), 0.99, row_h - 0.04,
            linewidth=0, facecolor=bg, zorder=1))

        for i, (val, xc) in enumerate(zip(row, xs_center)):
            fs = 12
            fw = 'normal'
            color = '#2C3E50'
            if i == 4 and val in highlight_avgs:
                color = '#1a7a40'
                fw = 'bold'
            ha = 'center' if i < 5 else 'left'
            x_pos = xc if i < 5 else xs[5] + 0.005
            ax.text(x_pos, y, val, ha=ha, va='center',
                    fontsize=fs, fontweight=fw, color=color,
                    fontproperties=font_prop, zorder=3)

    # border lines
    table_top = fig_h - top_pad
    table_bot = fig_h - top_pad - header_h - n_rows * row_h + 0.03
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.005, table_bot), 0.99, table_top - table_bot,
        boxstyle="round,pad=0.01", linewidth=1.2,
        edgecolor='#BDC3C7', facecolor='none', zorder=4))

    plt.tight_layout(pad=0.1)
    plt.savefig('/home/minji/autoresearch/table1_score_progress.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("table1 saved.")


# ─────────────────────────────────────────────
# TABLE 2: 실패 패턴 분석
# ─────────────────────────────────────────────
def make_table2():
    headers = ['실패 유형', '실험 수', '대표 실험', '원인']
    rows = [
        ['rs ↑ but s ↓ (불균형)', '8개',
         '33adaa5, 8f42694, 948af22\n9d13700, 0b11a7d, 81833df 등',
         'random_scaffold에선 좋지만 scaffold(hard split)에서\n점수 하락 — 과적합 또는 서치 공간 불균형'],
        ['HPO 예산 희석', '5개',
         'ab15bc3(stochastic depth)\n6439e21(Mixup), 0332106(pw_scale)',
         '파라미터가 너무 많아져 30 trials 내에\n최적값을 찾지 못함'],
        ['아키텍처와 충돌', '4개',
         'ebb514e(modality ID embed)\nc51d0b8(CLS token)',
         'SGU가 이미 위치 정보 처리 → ID 임베딩 중복\nCLS 토큰이 SGU 포지셔널 믹싱 방해'],
        ['LR 스케줄러 문제', '3개',
         'a1f88dd(cosine)\n8380d2b(cosine annealing)\nc0f90ad(ReduceLROnPlateau)',
         'Early stopping과 충돌 —\nLR이 너무 일찍 감소하거나 작은 모델로 수렴'],
        ['손실함수 변경 역효과', '2개',
         'a6ca1c9(focal loss)\n81833df(class weighting ×3)',
         'scaffold split에서 BBB- 예측 과도하게 강조 →\nscaffold 점수 급락'],
    ]

    col_widths = [0.22, 0.07, 0.28, 0.41]
    fig_w = 18
    row_h = 1.15
    header_h = 0.70
    top_pad = 0.9
    bottom_pad = 0.4

    n_rows = len(rows)
    fig_h = top_pad + header_h + n_rows * row_h + bottom_pad

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis('off')

    title_y = fig_h - 0.35
    ax.text(0.5, title_y, '실패 패턴 분석 (discard 실험)',
            ha='center', va='center', fontsize=18, fontweight='bold',
            fontproperties=font_prop)

    xs = []
    x = 0.01
    for w in col_widths:
        xs.append(x)
        x += w
    xs_center = [xs[i] + col_widths[i]/2 for i in range(len(col_widths))]

    hy = fig_h - top_pad - header_h/2
    for i, (h, xc) in enumerate(zip(headers, xs_center)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (xs[i]+0.002, hy - header_h/2 + 0.04),
            col_widths[i] - 0.004, header_h - 0.05,
            boxstyle="round,pad=0.01", linewidth=0,
            facecolor='#922B21', zorder=2))
        ax.text(xc, hy, h, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white',
                fontproperties=font_prop, zorder=3)

    row_colors = ['#FDEDEC', '#white', '#FDEDEC', 'white', '#FDEDEC']
    alt_colors = ['#FDEDEC', 'white']
    for r_idx, row in enumerate(rows):
        y = fig_h - top_pad - header_h - (r_idx + 0.5) * row_h
        bg = alt_colors[r_idx % 2]
        ax.add_patch(mpatches.Rectangle(
            (0.005, y - row_h/2 + 0.03), 0.99, row_h - 0.04,
            linewidth=0, facecolor=bg, zorder=1))

        for i, (val, xc) in enumerate(zip(row, xs_center)):
            ha = 'center' if i < 2 else 'left'
            x_pos = xc if i < 2 else xs[i] + 0.008
            fw = 'bold' if i == 0 else 'normal'
            ax.text(x_pos, y, val, ha=ha, va='center',
                    fontsize=11.5, fontweight=fw, color='#2C3E50',
                    fontproperties=font_prop, zorder=3,
                    multialignment='left' if i >= 2 else 'center')

    table_top = fig_h - top_pad
    table_bot = fig_h - top_pad - header_h - n_rows * row_h + 0.03
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.005, table_bot), 0.99, table_top - table_bot,
        boxstyle="round,pad=0.01", linewidth=1.2,
        edgecolor='#BDC3C7', facecolor='none', zorder=4))

    plt.tight_layout(pad=0.1)
    plt.savefig('/home/minji/autoresearch/table2_failure_patterns.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("table2 saved.")


# ─────────────────────────────────────────────
# TABLE 3: 파일 구조
# ─────────────────────────────────────────────
def make_table3():
    fig, axes = plt.subplots(3, 1, figsize=(17, 18))
    fig.patch.set_facecolor('white')

    # subtitle common style
    subtitle_kw = dict(fontsize=16, fontweight='bold', color='#1A252F',
                       fontproperties=font_prop, ha='center', va='center')

    # ── Section A: 원본 프레임워크 ──
    ax = axes[0]
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.93, '① 원본 autoresearch 프레임워크 (Karpathy)',
            transform=ax.transAxes, **subtitle_kw)

    headers = ['파일', '역할']
    rows_a = [
        ['prepare.py', '원본: GPT 학습용 데이터 준비 (고정)'],
        ['train.py', '원본: GPT 모델 + 학습 루프 (agent가 수정)'],
        ['program.md', '원본: agent 지시서 템플릿'],
        ['pyproject.toml / uv.lock', '의존성 관리 (uv 기반)'],
    ]
    col_w = [0.25, 0.73]
    draw_simple_table(ax, headers, rows_a, col_w,
                      header_color='#1A5276', y_start=0.82, row_h=0.16)

    # ── Section B: BBB용 파일 ──
    ax = axes[1]
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, '② BBB용으로 새로 준비한 파일',
            transform=ax.transAxes, **subtitle_kw)

    headers_b = ['파일', '수정 가능', '역할']
    rows_b = [
        ['bbb_prepare.py', '불가 (고정)', 'prepare.py의 BBB 버전 — 데이터 로딩, 8-모달리티 피처 엔지니어링, scaffold split, 평가 함수'],
        ['bbb_train.py', '가능 (agent 수정)', 'train.py의 BBB 버전 — gMLP 아키텍처 + Optuna HPO, agent가 이 파일만 수정'],
        ['bbb_program.md', '참조용', 'program.md의 BBB 버전 — 실험 루프 지시서 (무한 루프 방식)'],
        ['bbb_results.tsv', '기록 전용', '실험 결과 로그 (git commit 안 함)'],
        ['bbb_experiment_overview.md', '참조용', '전체 실험 구조 설명 문서'],
    ]
    col_w_b = [0.26, 0.18, 0.54]
    draw_simple_table(ax, headers_b, rows_b, col_w_b,
                      header_color='#1A5276', y_start=0.84, row_h=0.16,
                      special_col=1,
                      special_fn=lambda v: ('#C0392B' if '불가' in v else
                                            '#1a7a40' if '가능' in v else '#7D6608'))

    # ── Section C: 데이터 파일 ──
    ax = axes[2]
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.93, '③ 데이터 파일 (외부 경로)',
            transform=ax.transAxes, **subtitle_kw)

    headers_c = ['경로', '내용']
    rows_c = [
        ['/home/minji/BBB/scage/BBB/data/bench_label.csv', 'SMILES + BBB 레이블 (p_np 컬럼: BBB+=1, BBB-=0)'],
        ['/home/minji/BBB/scage/BBB/data/bench_embed.csv', 'scage 분자 수준 임베딩 (scage1)'],
        ['/home/minji/BBB/scage/BBB/data/bench_atom_embed.csv', 'scage 원자 수준 임베딩 (scage2)'],
        ['/home/minji/BBB/mole_public/MolE_embed_base_bbb.csv', 'MolE 언어모델 임베딩 (768차원)'],
    ]
    col_w_c = [0.50, 0.48]
    draw_simple_table(ax, headers_c, rows_c, col_w_c,
                      header_color='#1A5276', y_start=0.82, row_h=0.16)

    plt.suptitle('BBB Autoresearch — 파일 구조', fontsize=20, fontweight='bold',
                 fontproperties=font_prop, y=0.99, color='#1A252F')
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.5)
    plt.savefig('/home/minji/autoresearch/table3_file_structure.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("table3 saved.")


def draw_simple_table(ax, headers, rows, col_widths,
                      header_color='#2C3E50', y_start=0.9, row_h=0.14,
                      special_col=None, special_fn=None):
    xs = [0.01]
    for w in col_widths[:-1]:
        xs.append(xs[-1] + w)

    # header
    hy = y_start
    for i, (h, x, w) in enumerate(zip(headers, xs, col_widths)):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x+0.003, hy - row_h*0.45), w - 0.006, row_h*0.88,
            boxstyle="round,pad=0.005", linewidth=0,
            facecolor=header_color, zorder=2,
            transform=ax.transAxes))
        ax.text(x + w/2, hy, h, ha='center', va='center',
                fontsize=13, fontweight='bold', color='white',
                fontproperties=font_prop, zorder=3, transform=ax.transAxes)

    for r_idx, row in enumerate(rows):
        y = y_start - (r_idx + 1) * row_h
        bg = '#EAF2FF' if r_idx % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle(
            (0.008, y - row_h*0.45), 0.984, row_h*0.88,
            linewidth=0, facecolor=bg, zorder=1,
            transform=ax.transAxes))

        for i, (val, x, w) in enumerate(zip(row, xs, col_widths)):
            color = '#2C3E50'
            fw = 'normal'
            if special_col is not None and i == special_col and special_fn:
                color = special_fn(val)
                fw = 'bold'
            if i == 0:
                color = '#154360'
                fw = 'bold'
            ax.text(x + 0.012, y, val, ha='left', va='center',
                    fontsize=11, fontweight=fw, color=color,
                    fontproperties=font_prop, zorder=3,
                    transform=ax.transAxes)

    # outer border
    total_h = (len(rows) + 1) * row_h
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.008, y_start - total_h + row_h*0.45), 0.984, total_h,
        boxstyle="round,pad=0.005", linewidth=1,
        edgecolor='#BDC3C7', facecolor='none', zorder=4,
        transform=ax.transAxes))


make_table1()
make_table2()
make_table3()
print("All done.")
