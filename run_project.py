from src.config import PATH_IN, SHEET, OUT_DIR, THEME_TEXT, START_DATE, END_DATE
from src.data_ingest import load_universe, add_company_descriptions, download_prices
from src.signal_methods import add_tfidf_similarity, add_embeddings, add_embedding_similarity
from src.alpha_model import build_alpha
from src.backtest import build_all_weight_schedules, run_all_backtests
from src.metrics import compute_perf_stats


def main():
    # Load text data.
    df = load_universe(PATH_IN, SHEET)
    df = add_company_descriptions(df)
    df = df[df["Company Description"].astype(str).str.strip() != ""].copy()

    # Build TF-IDF signal and normalize it.
    df = add_tfidf_similarity(df, THEME_TEXT)
    df["alpha_tfidf"] = build_alpha(df["tfidf_similarity"])

    # Build embedding signal and normalize it.
    df = add_embeddings(df)
    df = add_embedding_similarity(df, THEME_TEXT)
    df["alpha_embedding"] = build_alpha(df["embedding_similarity"])

    df.to_pickle(OUT_DIR / "scored_universe.pkl")

    # Download prices and create target-weight schedules.
    ticker_list = df["Ticker"].tolist()
    prices = download_prices(ticker_list, START_DATE, END_DATE)
    
    #  Create new alpha dataframe for tfidf indexed by ticker name
    alpha_tfidf = df.set_index("Ticker")["alpha_tfidf"].dropna()
    
    #  Create new alpha data frame for tfidf indexed by ticker name
    alpha_embedding = df.set_index("Ticker")["alpha_embedding"].dropna()

    # Run weight schedules.
    schedules = build_all_weight_schedules(prices, alpha_tfidf, alpha_embedding)

    for name, schedule in schedules.items():
        schedule.to_csv(OUT_DIR / f"weights_{name}.csv")

    # Run backtests and save results.
    returns_df, values_df = run_all_backtests(prices, schedules)
    returns_df.to_csv(OUT_DIR / "returns_all_strategies.csv")
    values_df.to_csv(OUT_DIR / "portfolio_values_all_strategies.csv")

    # Print simple summaries.
    print("\nTop 10 names by TF-IDF:")
    print(df[["Ticker", "tfidf_similarity", "alpha_tfidf"]].sort_values("tfidf_similarity", ascending=False).head(10))

    print("\nTop 10 names by embeddings:")
    print(df[["Ticker", "embedding_similarity", "alpha_embedding"]].sort_values("embedding_similarity", ascending=False).head(10))

    print("\nPerformance summary:")
    for strategy_name in returns_df.columns:
        print(strategy_name, compute_perf_stats(returns_df[strategy_name].dropna()))


if __name__ == "__main__":
    main()
