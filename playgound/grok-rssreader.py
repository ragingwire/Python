#!/usr/bin/env python3
"""Scrollable RSS feed viewer — up to 5 feeds, refreshed every minute."""

import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from tkinter import font as tkfont
from tkinter import ttk

import feedparser

# Up to 5 feeds: (display name, URL, color)
RSS_FEEDS = [
    ("BBC News", "http://feeds.bbci.co.uk/news/rss.xml", "#2563eb"),
    ("NPR Top Stories", "https://feeds.npr.org/1001/rss.xml", "#dc2626"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index", "#16a34a"),
    ("Hacker News", "https://hnrss.org/frontpage", "#9333ea"),
    ("Reddit Technology", "https://www.reddit.com/r/technology/.rss", "#ea580c"),
]

REFRESH_INTERVAL_MS = 60_000
MAX_ITEMS_PER_FEED = 15
WINDOW_TITLE = "RSS Feed Viewer"


def entry_link(entry) -> str:
    """Return the best available link for a feed entry."""
    if getattr(entry, "link", None):
        return entry.link
    links = getattr(entry, "links", None) or []
    for link in links:
        href = link.get("href") if isinstance(link, dict) else getattr(link, "href", None)
        if href:
            return href
    return ""


def entry_title(entry) -> str:
    title = getattr(entry, "title", None) or "Untitled"
    return " ".join(str(title).split())


class RSSViewerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("900x600")
        self.root.minsize(500, 400)

        self._link_map: dict[str, str] = {}
        self._fetching = False
        self._headline_font = tkfont.Font(family="Segoe UI", size=11)
        self._title_font = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self._status_var = tk.StringVar(value="Loading feeds…")

        self._build_ui()
        self._schedule_refresh(immediate=True)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        status = ttk.Label(outer, textvariable=self._status_var, anchor=tk.W)
        status.pack(fill=tk.X, pady=(0, 6))

        canvas_frame = ttk.Frame(outer)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame,
            highlightthickness=0,
            background="#f8fafc",
        )
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.content = ttk.Frame(self.canvas)
        self._canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor=tk.NW)

        self.content.bind("<Configure>", self._on_content_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self._bind_mousewheel(self.canvas)
        self._bind_mousewheel(self.content)

    def _bind_mousewheel(self, widget: tk.Widget) -> None:
        widget.bind("<MouseWheel>", self._on_mousewheel_vertical)
        widget.bind("<Shift-MouseWheel>", self._on_mousewheel_horizontal)
        widget.bind("<Button-4>", self._on_mousewheel_vertical_linux)
        widget.bind("<Button-5>", self._on_mousewheel_vertical_linux)
        widget.bind("<Shift-Button-4>", self._on_mousewheel_horizontal_linux)
        widget.bind("<Shift-Button-5>", self._on_mousewheel_horizontal_linux)

    def _on_content_configure(self, event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if event is not None:
            self.canvas.itemconfig(
                self._canvas_window,
                width=max(self.canvas.winfo_width(), event.width),
            )

    def _on_canvas_configure(self, event) -> None:
        # Grow with wide headlines; never shrink content below the viewport width.
        self.canvas.itemconfig(
            self._canvas_window,
            width=max(event.width, self.content.winfo_reqwidth()),
        )

    def _on_mousewheel_vertical(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_mousewheel_horizontal(self, event) -> None:
        self.canvas.xview_scroll(int(-event.delta / 120), "units")

    def _on_mousewheel_vertical_linux(self, event) -> None:
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")

    def _on_mousewheel_horizontal_linux(self, event) -> None:
        if event.num == 4:
            self.canvas.xview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.xview_scroll(1, "units")

    def _schedule_refresh(self, immediate: bool = False) -> None:
        if immediate:
            self._start_fetch()
        self.root.after(REFRESH_INTERVAL_MS, self._refresh_cycle)

    def _refresh_cycle(self) -> None:
        self._start_fetch()
        self.root.after(REFRESH_INTERVAL_MS, self._refresh_cycle)

    def _start_fetch(self) -> None:
        if self._fetching:
            return
        self._fetching = True
        self._status_var.set("Updating feeds…")
        thread = threading.Thread(target=self._fetch_all_feeds, daemon=True)
        thread.start()

    def _fetch_all_feeds(self) -> None:
        results = []
        for name, url, color in RSS_FEEDS[:5]:
            try:
                parsed = feedparser.parse(url)
                entries = []
                for entry in parsed.entries[:MAX_ITEMS_PER_FEED]:
                    link = entry_link(entry)
                    if link:
                        entries.append((entry_title(entry), link))
                feed_title = getattr(parsed.feed, "title", None) or name
                results.append(
                    {
                        "name": name,
                        "feed_title": feed_title,
                        "color": color,
                        "entries": entries,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — show fetch errors in UI
                results.append(
                    {
                        "name": name,
                        "feed_title": name,
                        "color": color,
                        "entries": [],
                        "error": str(exc),
                    }
                )

        self.root.after(0, lambda: self._render_results(results))

    def _clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        self._link_map.clear()

    def _render_results(self, results: list[dict]) -> None:
        self._clear_content()
        tag_counter = 0

        for feed in results:
            color = feed["color"]
            header = tk.Label(
                self.content,
                text=feed["feed_title"],
                font=self._title_font,
                fg=color,
                bg="#f8fafc",
                anchor=tk.W,
                justify=tk.LEFT,
            )
            header.pack(fill=tk.X, padx=4, pady=(14, 4))

            if feed["error"]:
                err = tk.Label(
                    self.content,
                    text=f"Could not load feed: {feed['error']}",
                    font=self._headline_font,
                    fg="#64748b",
                    bg="#f8fafc",
                    anchor=tk.W,
                )
                err.pack(fill=tk.X, padx=12, pady=2)
                continue

            if not feed["entries"]:
                empty = tk.Label(
                    self.content,
                    text="No headlines available.",
                    font=self._headline_font,
                    fg="#64748b",
                    bg="#f8fafc",
                    anchor=tk.W,
                )
                empty.pack(fill=tk.X, padx=12, pady=2)
                continue

            for title, link in feed["entries"]:
                tag = f"headline_{tag_counter}"
                tag_counter += 1
                self._link_map[tag] = link

                row = tk.Label(
                    self.content,
                    text=f"  • {title}",
                    font=self._headline_font,
                    fg=color,
                    bg="#f8fafc",
                    anchor=tk.W,
                    justify=tk.LEFT,
                    cursor="hand2",
                )
                row.pack(anchor=tk.W, padx=12, pady=2)
                row.bind("<Button-1>", lambda _e, t=tag: self._open_link(t))
                row.bind("<Enter>", lambda _e, w=row, c=color: w.configure(fg="#000000"))
                #row.bind("<Enter>", lambda _e, w=row, c=color: w.configure(fg=self._hover_color(c)))
                row.bind("<Leave>", lambda _e, w=row, c=color: w.configure(fg=c))

        updated = datetime.now().strftime("%H:%M:%S")
        total = sum(len(f["entries"]) for f in results)
        self._status_var.set(f"Last updated {updated} — {total} headlines from {len(results)} feeds")
        self._fetching = False
        self._on_content_configure()

    @staticmethod
    def _hover_color(hex_color: str) -> str:
        """Slightly darken a hex color for hover feedback without underlines."""
        hex_color = hex_color.lstrip("#")
        r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        factor = 0.82
        return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"

    def _open_link(self, tag: str) -> None:
        url = self._link_map.get(tag)
        if url:
            webbrowser.open(url)


def main() -> None:
    root = tk.Tk()
    RSSViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()