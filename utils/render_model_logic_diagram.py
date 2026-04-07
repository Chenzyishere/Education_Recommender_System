import os

from PIL import Image, ImageDraw, ImageFont


def load_font(size):
    """Load a Chinese-capable font on Windows, fallback to default."""
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_multiline_center(draw, rect, text, font, fill="#1A202C", line_spacing=10):
    """Draw multiline centered text inside a rectangle."""
    x1, y1, x2, y2 = rect
    lines = str(text).split("\n")
    sizes = [draw.textsize(line, font=font) for line in lines]
    text_w = max((w for w, _ in sizes), default=0)
    text_h = sum((h for _, h in sizes)) + max(0, len(lines) - 1) * line_spacing
    start_x = x1 + (x2 - x1 - text_w) / 2
    y = y1 + (y2 - y1 - text_h) / 2
    for line, (w, h) in zip(lines, sizes):
        x = x1 + (x2 - x1 - w) / 2
        draw.text((x, y), line, font=font, fill=fill)
        y += h + line_spacing


def draw_box(draw, rect, text, fill, outline, font):
    """Draw a box and centered text."""
    draw.rectangle(rect, fill=fill, outline=outline, width=4)
    draw_multiline_center(draw, rect, text, font)


def draw_arrow(draw, start, end, color="#2D3748", width=6, head=16):
    """Draw a directional arrow."""
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        if x2 >= x1:
            points = [(x2, y2), (x2 - head, y2 - head // 2), (x2 - head, y2 + head // 2)]
        else:
            points = [(x2, y2), (x2 + head, y2 - head // 2), (x2 + head, y2 + head // 2)]
    else:
        if y2 >= y1:
            points = [(x2, y2), (x2 - head // 2, y2 - head), (x2 + head // 2, y2 - head)]
        else:
            points = [(x2, y2), (x2 - head // 2, y2 + head), (x2 + head // 2, y2 + head)]
    draw.polygon(points, fill=color)


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rendered_dir = os.path.join(base_dir, "rendered")
    os.makedirs(rendered_dir, exist_ok=True)
    save_path = os.path.join(rendered_dir, "kg_sakt_logic_diagram.png")

    # Vertical layout, no title.
    width, height = 1700, 2400
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(38)

    # Main vertical flow (strict causal order).
    b1 = (320, 80, 1380, 320)     # 输入层
    b2 = (320, 400, 1380, 640)    # 认知建模层
    b3 = (320, 720, 1380, 960)    # 候选生成
    bk = (320, 1040, 1380, 1280)  # 知识图谱层
    bl = (320, 1360, 1380, 1600)  # 逻辑约束层
    b4 = (320, 1680, 1380, 1920)  # 融合排序与决策
    be = (320, 2020, 1380, 2320)  # 解释输出

    draw_box(
        draw,
        b1,
        "输入层\n学生交互序列\n（skill_id、correct、order_id、sequence_idx）",
        fill="#EBF8FF",
        outline="#2B6CB0",
        font=font,
    )
    draw_box(
        draw,
        b2,
        "认知建模层\nSAKT 因果自注意力编码",
        fill="#E6FFFA",
        outline="#2B6CB0",
        font=font,
    )
    draw_box(
        draw,
        b3,
        "掌握概率估计\nP(mastery | history)\n候选资源生成",
        fill="#F0FFF4",
        outline="#2B6CB0",
        font=font,
    )
    draw_box(
        draw,
        bk,
        "知识图谱层\nkg_triples.json\nGraph={kp:[prerequisites]}",
        fill="#FAF5FF",
        outline="#6B46C1",
        font=font,
    )
    draw_box(
        draw,
        bl,
        "逻辑约束层\n先修可达性校验\n路径一致性检查",
        fill="#F7FAFC",
        outline="#4A5568",
        font=font,
    )
    draw_box(
        draw,
        b4,
        "融合排序与决策\n掌握概率 + ZPD + 逻辑约束\nTop-K 推荐知识点",
        fill="#FFF5F5",
        outline="#2B6CB0",
        font=font,
    )
    draw_box(
        draw,
        be,
        "解释输出\n因为已掌握前置知识 A/B\n所以推荐学习 C",
        fill="#FFFAF0",
        outline="#C05621",
        font=font,
    )

    # Strict sequential arrows: 候选 -> 图谱 -> 约束 -> 排序 -> 解释
    draw_arrow(draw, (850, 320), (850, 400))
    draw_arrow(draw, (850, 640), (850, 720))
    draw_arrow(draw, (850, 960), (850, 1040))
    draw_arrow(draw, (850, 1280), (850, 1360))
    draw_arrow(draw, (850, 1600), (850, 1680))
    draw_arrow(draw, (850, 1920), (850, 2020))

    img.save(save_path, format="PNG")
    print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()
