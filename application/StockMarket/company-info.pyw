import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import threading
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates

class CompanyDataFetcher:
    """
    Handles fetching and formatting financial data from the yfinance API.
    """
    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.raw_info = {}

    def fetch_info(self):
        """Fetches data from yfinance. Returns True if successful, False otherwise."""
        try:
            # The .info attribute fetches a dictionary of company data
            self.raw_info = self.ticker.info
            
            # yfinance sometimes returns a dictionary with just a 'trailingPegRatio' 
            # if the ticker is completely invalid. We verify a standard key exists.
            if 'shortName' not in self.raw_info and 'logo_url' not in self.raw_info:
                return False
                
            return True
        except Exception as e:
            self.raw_info = {"error": str(e)}
            return False

    def fetch_history(self):
        """Fetches 1-day intraday history data (1-minute intervals)."""
        try:
            hist = self.ticker.history(period="1d", interval="1m")
            if not hist.empty:
                return hist
            return None
        except Exception:
            return None

    def get_organized_data(self):
        """Organizes the raw yfinance dictionary into logical categories."""
        if not self.raw_info:
            return {}

        data = {
            "General Information": {
                "Company Name": self.raw_info.get("shortName", "N/A"),
                "Sector": self.raw_info.get("sector", "N/A"),
                "Industry": self.raw_info.get("industry", "N/A"),
                "Country": self.raw_info.get("country", "N/A"),
                "Website": self.raw_info.get("website", "N/A"),
                "Employees": self._format_number(self.raw_info.get("fullTimeEmployees")),
                "Currency": self.raw_info.get("financialCurrency", "N/A")
            },
            "Financial Metrics": {
                "Market Cap": self._format_currency(self.raw_info.get("marketCap")),
                "Total Revenue": self._format_currency(self.raw_info.get("totalRevenue")),
                "EBITDA": self._format_currency(self.raw_info.get("ebitda")),
                "Trailing P/E": self._format_decimal(self.raw_info.get("trailingPE")),
                "Forward P/E": self._format_decimal(self.raw_info.get("forwardPE")),
                "Profit Margin": self._format_percentage(self.raw_info.get("profitMargins")),
                "Operating Margin": self._format_percentage(self.raw_info.get("operatingMargins")),
                "Dividend Yield": self._format_percentage(self.raw_info.get("dividendYield")),
                "52 Week High": self._format_decimal(self.raw_info.get("fiftyTwoWeekHigh")),
                "52 Week Low": self._format_decimal(self.raw_info.get("fiftyTwoWeekLow"))
            },
            "Business Summary": self.raw_info.get("longBusinessSummary", "No business summary available.")
        }
        return data

    # --- Formatting Helper Methods ---
    def _format_currency(self, value):
        if value is None or value == "N/A": return "N/A"
        return f"${value:,.0f}"

    def _format_number(self, value):
        if value is None or value == "N/A": return "N/A"
        return f"{value:,}"

    def _format_decimal(self, value):
        if value is None or value == "N/A": return "N/A"
        return f"{value:.2f}"

    def _format_percentage(self, value):
        if value is None or value == "N/A": return "N/A"
        return f"{value * 100:.2f}%"


class CompanyInfoApp(tk.Tk):
    """
    Main Application class for the GUI, inheriting from tk.Tk.
    """
    def __init__(self):
        super().__init__()
        
        # Window configuration
        self.title("Stock & Company Information Explorer")
        self.geometry("750x550")
        self.configure(padx=15, pady=15)
        
        # Set a modern ttk theme if available
        style = ttk.Style(self)
        if 'clam' in style.theme_names():
            style.theme_use('clam')
            
        self.refresh_job = None
        self.current_ticker = None
            
        self.create_widgets()

    def create_widgets(self):
        """Builds the GUI layout."""
        # --- Top Search Bar ---
        search_frame = ttk.Frame(self)
        search_frame.pack(fill=tk.X, pady=(0, 15))

        ttk.Label(search_frame, text="Ticker Symbol (e.g., AAPL, TSLA):", font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.ticker_entry = ttk.Entry(search_frame, width=20, font=("Helvetica", 11))
        self.ticker_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.ticker_entry.bind("<Return>", lambda event: self.start_search()) # Bind Enter key
        
        self.search_btn = ttk.Button(search_frame, text="Search", command=self.start_search)
        self.search_btn.pack(side=tk.LEFT)
        
        self.progress = ttk.Progressbar(search_frame, mode='indeterminate')

        # --- Data Display Notebook (Tabs) ---
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create frames for tabs
        self.general_tab = ttk.Frame(self.notebook, padding=10)
        self.financials_tab = ttk.Frame(self.notebook, padding=10)
        self.summary_tab = ttk.Frame(self.notebook, padding=10)
        self.chart_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.general_tab, text="General Info")
        self.notebook.add(self.financials_tab, text="Financials")
        self.notebook.add(self.summary_tab, text="Business Summary")
        self.notebook.add(self.chart_tab, text="Live Chart")

        # Setup Treeviews for structured key-value data
        self.general_tree = self._setup_treeview(self.general_tab)
        self.financials_tree = self._setup_treeview(self.financials_tab)

        # Setup Text area for the long business summary
        self.summary_text = tk.Text(self.summary_tab, wrap=tk.WORD, state=tk.DISABLED, 
                                    font=("Helvetica", 11), bg="#f9f9f9")
        scrollbar = ttk.Scrollbar(self.summary_tab, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=scrollbar.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Setup Matplotlib Figure for the Chart Tab
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.chart_tab)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _setup_treeview(self, parent):
        """Helper to create a standard 2-column Treeview."""
        tree = ttk.Treeview(parent, columns=("Attribute", "Value"), show="headings")
        tree.heading("Attribute", text="Attribute")
        tree.heading("Value", text="Value")
        
        tree.column("Attribute", width=200, anchor=tk.W)
        tree.column("Value", width=450, anchor=tk.W)
        
        # Add alternating row colors
        tree.tag_configure('evenrow', background='#f0f0ff')
        tree.tag_configure('oddrow', background='#ffffff')
        
        tree.pack(fill=tk.BOTH, expand=True)
        return tree

    def start_search(self):
        """Triggered when the user clicks Search or presses Enter."""
        ticker = self.ticker_entry.get().strip()
        if not ticker:
            messagebox.showwarning("Input Error", "Please enter a stock ticker symbol.")
            return

        # Cancel any pending auto-refresh jobs
        if self.refresh_job is not None:
            self.after_cancel(self.refresh_job)
            self.refresh_job = None
            
        self.current_ticker = ticker

        # UI Updates during loading
        self.search_btn.config(state=tk.DISABLED)
        self.progress.pack(side=tk.LEFT, padx=15)
        self.progress.start()
        
        self._clear_data()

        # Execute network request in a separate thread so GUI doesn't freeze
        threading.Thread(target=self.fetch_data_thread, args=(ticker,), daemon=True).start()

    def fetch_data_thread(self, ticker):
        """Runs in background thread to fetch data."""
        fetcher = CompanyDataFetcher(ticker)
        success = fetcher.fetch_info()
        hist_data = fetcher.fetch_history() if success else None
        
        # Schedule the UI update back on the main thread
        self.after(0, self.update_gui_callback, success, fetcher, hist_data)

    def update_gui_callback(self, success, fetcher, hist_data):
        """Runs on the main thread to update widgets with fetched data."""
        self.progress.stop()
        self.progress.pack_forget()
        self.search_btn.config(state=tk.NORMAL)

        if not success:
            messagebox.showerror("Not Found", 
                                 f"Failed to find data for ticker '{fetcher.ticker_symbol}'.\n"
                                 "Please verify the symbol is correct and you have internet access.")
            return

        data = fetcher.get_organized_data()

        # Populate General Info Treeview
        for idx, (key, value) in enumerate(data["General Information"].items()):
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            self.general_tree.insert("", tk.END, values=(key, value), tags=(tag,))

        # Populate Financials Treeview
        for idx, (key, value) in enumerate(data["Financial Metrics"].items()):
            tag = 'evenrow' if idx % 2 == 0 else 'oddrow'
            self.financials_tree.insert("", tk.END, values=(key, value), tags=(tag,))

        # Populate Business Summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.insert(tk.END, data["Business Summary"])
        self.summary_text.config(state=tk.DISABLED)

        # Update Chart
        self.update_chart(hist_data, fetcher.ticker_symbol)

        # Schedule next auto-refresh for the chart (every 5 seconds)
        self.refresh_job = self.after(5000, self.trigger_chart_refresh)

    def update_chart(self, hist_data, ticker_symbol):
        """Draws the matplotlib chart with historical data."""
        self.ax.clear()
        
        if hist_data is None or hist_data.empty:
            self.ax.text(0.5, 0.5, "No intraday data available currently.", 
                         ha='center', va='center', transform=self.ax.transAxes)
        else:
            times = hist_data.index
            prices = hist_data['Close']
            high_price = hist_data['High'].max()
            low_price = hist_data['Low'].min()

            # Plot Closing Price
            self.ax.plot(times, prices, color='#0078D7', linewidth=1.5, label='Close Price')
            
            # Plot High and Low lines
            self.ax.axhline(high_price, color='#107C10', linestyle='--', alpha=0.7, label=f'High: ${high_price:.2f}')
            self.ax.axhline(low_price, color='#D83B01', linestyle='--', alpha=0.7, label=f'Low: ${low_price:.2f}')

            # Plot crosshairs for the most actual price
            latest_time = times[-1]
            latest_price = prices.iloc[-1]
            self.ax.axhline(latest_price, color='black', linestyle=':', alpha=0.5)
            self.ax.axvline(latest_time, color='black', linestyle=':', alpha=0.5)
            self.ax.plot(latest_time, latest_price, marker='o', color='red', markersize=4, label=f'Current: ${latest_price:.2f}')

            # Formatting
            self.ax.set_title(f"{ticker_symbol} - 1 Day Intraday History")
            self.ax.set_ylabel("Price (USD)")
            self.ax.legend(loc="upper left")
            
            # Format X-axis for time
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            self.figure.autofmt_xdate(rotation=45)
            self.ax.grid(True, linestyle=':', alpha=0.6)

        self.canvas.draw()

    def trigger_chart_refresh(self):
        """Triggers a background fetch just for the chart data to keep it realtime."""
        if not self.current_ticker:
            return
        threading.Thread(target=self.refresh_chart_thread, args=(self.current_ticker,), daemon=True).start()

    def refresh_chart_thread(self, ticker):
        """Background thread to fetch new chart data."""
        fetcher = CompanyDataFetcher(ticker)
        hist_data = fetcher.fetch_history()
        
        # Return to main thread to update chart
        self.after(0, self.refresh_chart_callback, hist_data, ticker)

    def refresh_chart_callback(self, hist_data, ticker):
        """Main thread callback to redraw chart and schedule the next tick."""
        # Only update if the user hasn't switched to a different ticker in the meantime
        if self.current_ticker == ticker:
            self.update_chart(hist_data, ticker)
            self.refresh_job = self.after(5000, self.trigger_chart_refresh)

    def _clear_data(self):
        """Clears all UI elements for a new search."""
        for item in self.general_tree.get_children(): 
            self.general_tree.delete(item)
            
        for item in self.financials_tree.get_children(): 
            self.financials_tree.delete(item)
            
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.config(state=tk.DISABLED)
        
        self.ax.clear()
        self.canvas.draw()

if __name__ == "__main__":
    app = CompanyInfoApp()
    app.mainloop()