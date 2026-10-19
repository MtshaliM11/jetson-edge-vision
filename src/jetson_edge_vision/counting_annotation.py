"""Compact counting overlay; detection/tracking rendering remains shared."""


def annotate_counting(frame, counter):
    """Draw in-place on the already copied annotated frame, bounded to 1/3 height.

    Reports include every category. The overlay shows as many rows as fit and
    explicitly indicates omitted categories on crowded/small frames.
    """
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (round(v) for v in counter.pixel_line)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 180, 255), 2, cv2.LINE_AA)
    totals = counter.totals()
    rows = [f"Crossings: {totals['total_crossings']}",
            f"A->B: {totals['by_direction']['A->B']}  B->A: {totals['by_direction']['B->A']}"]
    categories = list(totals["by_category"].values())
    scale = min(.55, max(.25, w / 1000))
    row_height = max(12, round(28 * scale))
    capacity = max(2, (h // 3 - 8) // row_height)
    available = max(0, capacity - 2)
    shown = len(categories) if len(categories) <= available else max(0, available - 1)
    for category in categories[:shown]:
        counts = category["by_direction"]
        rows.append(f"{category['class_name']}: A->B {counts['A->B']}  B->A {counts['B->A']}")
    if shown < len(categories) and available:
        rows.append(f"+{len(categories)-shown} categories in JSON")
    # Fit width as well as height; all visible numbers come from the same totals.
    longest = max(cv2.getTextSize(row, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] for row in rows)
    scale *= min(1., max(1, w-12) / max(1, longest))
    panel_width = min(w-1, longest+12)
    panel_left = max(0, w - panel_width - 1)
    cv2.rectangle(frame, (panel_left, 0), (w-1, min(h-1, len(rows)*row_height+6)), (20, 20, 20), -1)
    for i, row in enumerate(rows):
        cv2.putText(frame, row, (panel_left+5, (i+1)*row_height), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    return frame
