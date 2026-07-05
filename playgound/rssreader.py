#!/usr/bin/env python3
"""
RSS Feed Reader - a simple desktop GUI application.

Displays entries from up to 5 RSS feeds in a single scrollable window.
Each feed is shown in its own color. Feeds are automatically re-fetched
and refreshed once every minute.

Requirements:
    pip install feedparser

Usage:
    python rss_reader.py

Configuration:
    Edit the FEEDS list below to set the RSS feed URLs you want to follow
    (up to 5 entries). Each feed gets a color automatically assigned from
    the COLORS list, in order.
"""

import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from datetime import datetime

import feedparser
import webbrowser
import re

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Add up to 5 RSS feed URLs here.
FEEDS = [
    "https://www.welt.de/?service=Rss",
    "https://www.bild.de/feed/news.xml",
    "https://www.welt.de/feeds/section/politik.rss",
    "https://www.deraktionaer.de/aktionaer-news.rss",
    "https://www.wallstreet-online.de/rss/nachrichten-alle.xml",
]

# One color per feed slot, in the same order as FEEDS.
COLORS = [
    "#1fAABB",  # blue
    "#d62728",  # red
    "#FFAA00",  # green
    "#94FF88",  # purple
    "#ff00FF",  # orange
]

REFRESH_INTERVAL_MS = 300 * 1000  # 3 minutes
MAX_ENTRIES_PER_FEED = 12        # how many items to show per feed each refresh


class RSSReaderApp:
    def __init__(self ):
        self.root = tk.Tk()
        self.root.title("Python RSS Feed Reader")
        self.root.geometry("900x650")
        self.root.configure(bg="#1e1e1e")

        self.feeds = FEEDS[:5]  # enforce max of 5
        self.colors = COLORS[: len(self.feeds)]

        self._build_ui()
        self._start_refresh_loop()
        self.__run__()

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        header = tk.Frame(self.root, bg="#1e1e1e")
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 0))

        title_font = tkfont.Font(family="Helvetica", size=10, weight="normal")
        tk.Label(
            header, text="Python RSS Feed Reader", font=title_font,
            bg="#1e1e1e", fg="white"
        ).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Loading feeds...")
        tk.Label(
            header, textvariable=self.status_var, bg="#1e1e1e",
            fg="#aaaaaa", font=("Helvetica", 10)
        ).pack(side=tk.RIGHT)

        # Legend showing which color belongs to which feed
        legend = tk.Frame(self.root, bg="#1e1e1e")
        legend.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(4, 6))
        self._build_legend(legend)

        # Scrollable text area (vertical AND horizontal scrolling).
        # wrap=NONE means long lines are not wrapped, so a horizontal
        # scrollbar is needed to read lines that run off the right edge.
        body = tk.Frame(self.root, bg="#1e1e1e")
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        v_scroll = tk.Scrollbar(body, orient=tk.VERTICAL)
        h_scroll = tk.Scrollbar(body, orient=tk.HORIZONTAL)

        self.text = tk.Text(
            body, wrap=tk.NONE, bg="#252526", fg="#e0e0e0",
            insertbackground="white", font=("Helvetica", 10),
            padx=10, pady=10, state=tk.DISABLED, cursor="arrow",
            yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set
        )

        v_scroll.config(command=self.text.yview)
        h_scroll.config(command=self.text.xview)

        # Grid layout so both scrollbars sit flush against the text widget.
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=1)
        self.text.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        # Mouse wheel support: vertical scroll with the wheel, horizontal
        # scroll with Shift+wheel (standard convention on most platforms).
        self.text.bind("<MouseWheel>", self._on_mousewheel_v)      # Windows/macOS
        self.text.bind("<Shift-MouseWheel>", self._on_mousewheel_h)
        self.text.bind("<Button-4>", self._on_mousewheel_v)        # Linux scroll up
        self.text.bind("<Button-5>", self._on_mousewheel_v)        # Linux scroll down

        # Tag configuration: one tag per feed for the title color,
        # plus generic tags for metadata / body text.
        for i, color in enumerate(self.colors):
            self.text.tag_configure(f"feed{i}_title", foreground=color,
                                     font=("Helvetica", 11, "bold"))
            self.text.tag_configure(f"feed{i}_source", foreground=color,
                                     font=("Helvetica", 9, "bold"))
        self.text.tag_configure("meta", foreground="#888888",
                                 font=("Helvetica", 9, "italic"))
        self.text.tag_configure("summary", foreground="#cccccc",
                                 font=("Helvetica", 10))
        self.text.tag_configure("separator", foreground="#444444")

    def _build_legend(self, parent):
        for i, url in enumerate(self.feeds):
            swatch = tk.Frame(parent, bg=self.colors[i], width=14, height=14)
            swatch.pack(side=tk.LEFT, padx=(0 if i == 0 else 12, 4), pady=2)
            swatch.pack_propagate(False)
            label = self._short_name(url)
            tk.Label(parent, text=label, bg="#1e1e1e", fg=self.colors[i],
                     font=("Helvetica", 9, "bold")).pack(side=tk.LEFT)

    def _on_mousewheel_v(self, event):
        # Linux sends Button-4/5 events instead of a delta value.
        if event.num == 4:
            self.text.yview_scroll(-1, "units")
        elif event.num == 5:
            self.text.yview_scroll(1, "units")
        else:
            delta = -1 if event.delta > 0 else 1
            self.text.yview_scroll(delta * 10, "units")
        return "break"

    def _on_mousewheel_h(self, event):
        delta = -1 if event.delta > 0 else 1
        self.text.xview_scroll(delta, "units")
        return "break"

    @staticmethod
    def _short_name(url):
        # Derive a readable short label from the feed URL's domain.
        try:
            domain = url.split("//")[-1].split("/")[0]
            domain = domain.replace("www.", "").replace("feeds.", "")
            return domain
        except Exception:
            return url

    # -----------------------------------------------------------------
    # Refresh loop
    # -----------------------------------------------------------------
    def _start_refresh_loop(self):
        self._refresh_now()
        self.root.after(REFRESH_INTERVAL_MS, self._start_refresh_loop)

    def _refresh_now(self):
        self.status_var.set("Refreshing...")
        thread = threading.Thread(target=self._fetch_all_feeds, daemon=True)
        thread.start()

    def _fetch_all_feeds(self):
        """Runs in a background thread so the GUI never freezes."""
        results = []
        for i, url in enumerate(self.feeds):
            try:
                parsed = feedparser.parse(url)
                feed_title = parsed.feed.get("title", self._short_name(url))
                entries = parsed.entries[:MAX_ENTRIES_PER_FEED]
                results.append((i, feed_title, entries, None))
            except Exception as exc:
                results.append((i, self._short_name(url), [], str(exc)))

        # Hand results back to the main thread for safe UI updates.
        self.root.after(0, self._render_results, results)

    # -----------------------------------------------------------------
    # Rendering
    # -----------------------------------------------------------------
    def _render_results(self, results):
        self.text.configure(state=tk.NORMAL)
        self.text.delete("1.0", tk.END)

        for i, feed_title, entries, error in results:
            self.text.insert(tk.END, f"{feed_title}\n", f"feed{i}_source")

            if error:
                self.text.insert(tk.END, f"  (could not load feed: {error})\n",
                                  "meta")
            elif not entries:
                self.text.insert(tk.END, "  (no entries found)\n", "meta")
            else:
                for entry in entries:
                    title = entry.get("title", "(untitled)")
                    published = entry.get("published", entry.get("updated", ""))
                    link = entry.get("link", "")
                    summary = entry.get("summary", "")
                    # Strip simple HTML tags from summary for readability
                    summary = self._strip_html(summary)
                    if len(summary) > 220:
                        summary = summary[:220].rstrip() + "..."

                    self.text.insert(tk.END, "  \u2022 ", f"feed{i}_title")
                    title_start = self.text.index(tk.INSERT)
                    self.text.insert(tk.END, f"{title}\n", f"feed{i}_title")
                    if link:
                        self._bind_link_click(title_start, link)
                    meta_line = f"    {published}" if published else ""
                    if meta_line:
                        self.text.insert(tk.END, meta_line + "\n", "meta")
                    if summary:
                        self.text.insert(tk.END, f"    {summary}\n", "summary")
            self.text.insert(tk.END, ("-" * 100) + "\n\n", "separator")

        self.text.configure(state=tk.DISABLED)
        now = datetime.now().strftime("%H:%M:%S")
        self.status_var.set(f"Last updated at {now} • next refresh in 5 min")

    def _bind_link_click(self, index, url):
        tag_name = f"link_{index.replace('.', '_')}"
        end_index = f"{index} lineend"
        self.text.tag_add(tag_name, index, end_index)
        self.text.tag_bind(tag_name, "<Button-1>",
                            lambda e, u=url: self._open_link(u))
        self.text.tag_bind(tag_name, "<Enter>",
                            lambda e: self.text.config(cursor="hand2"))
        self.text.tag_bind(tag_name, "<Leave>",
                            lambda e: self.text.config(cursor="arrow"))

    @staticmethod
    def _open_link(url):
        webbrowser.open(url)

    @staticmethod
    def _strip_html(raw_html):
        text = re.sub(r"<[^>]+>", "", raw_html or "")
        return " ".join(text.split())
    
    def __run__(self):
        self.root.mainloop () 


if __name__ == "__main__":
    RSSReaderApp ()