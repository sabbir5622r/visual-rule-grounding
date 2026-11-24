from PIL import Image, ImageDraw, ImageFont


def draw_chart(rows, layout, destination):
    image = Image.new("RGB", (640, 420), "#ffffff")
    draw = ImageDraw.Draw(image)
    title = ImageFont.load_default(size=30)
    normal = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=18)
    ink = "#142637"
    first_color = "#17698c"
    second_color = "#a84c20"
    draw.text((44, 28), "Measurements", font=title, fill=ink)
    if layout == "table":
        for x, label in ((70, "Entry"), (315, "A"), (490, "B")):
            draw.text((x, 99), label, font=normal, fill=ink)
        for index, (entry, values) in enumerate(rows.items()):
            y = 145 + index * 74
            fill = "#f0f5f7" if index % 2 == 0 else "#f9fafb"
            draw.rounded_rectangle((45, y, 590, y + 59), radius=8, fill=fill)
            for x, label in zip((70, 315, 490), (entry, *values)):
                draw.text((x, y + 17), str(label), font=normal, fill=ink)
    else:
        for x, label, color in ((445, "A", first_color), (512, "B", second_color)):
            draw.rectangle((x, 45, x + 18, 57), fill=color)
            draw.text((x + 22, 40), label, font=small, fill=ink)
        for index, (entry, values) in enumerate(rows.items()):
            y = 93 + index * 91
            draw.text((56, y + 22), entry, font=normal, fill=ink)
            for offset, value, color in zip((0, 28), values, (first_color, second_color)):
                x = 140 + 4 * value
                draw.rectangle((140, y + offset, x, y + offset + 19), fill=color)
                draw.text((x + 10, y + offset - 2), str(value), font=small, fill=ink)
        draw.line((140, 368, 540, 368), fill="#61717e", width=2)
        for tick in (0, 25, 50, 75, 100):
            x = 140 + 4 * tick
            draw.line((x, 366, x, 375), fill="#61717e", width=2)
            draw.text((x, 381), str(tick), font=small, fill=ink, anchor="mt")
    image.save(destination, optimize=True)
