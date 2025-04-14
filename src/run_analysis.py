from pathlib import Path

from matplotlib import pyplot
import numpy as np

from political_party_analysis.loader import DataLoader
from political_party_analysis.visualization import scatter_plot, plot_density_estimation_results, plot_finnish_parties
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator

if __name__ == "__main__":

    data_loader = DataLoader()

    # Data pre-processing step
    # پردازش داده‌ها با استفاده از کلاس DataLoader
    processed_data = data_loader.preprocess_data()
    print(f"Processed data shape: {processed_data.shape}")

    # Dimensionality reduction step
    # کاهش ابعاد داده‌ها با استفاده از PCA
    dim_reducer = DimensionalityReducer(data=processed_data)
    reduced_dim_data = dim_reducer.transform()
    print(f"Reduced dimension data shape: {reduced_dim_data.shape}")

    # Uncomment this snippet to plot dim reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="dim reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    # Density estimation/distribution modelling step
    # تخمین چگالی با استفاده از مدل‌های مخلوط گاوسی
    # حفظ نام‌های ویژگی‌های داده‌های اصلی برای استفاده در تبدیل معکوس
    feature_names = processed_data.columns

    density_estimator = DensityEstimator(data=reduced_dim_data, dim_reducer=dim_reducer, high_dim_feature_names=feature_names)

    # برازش مدل و دریافت برچسب‌ها، میانگین‌ها و کوواریانس‌ها
    labels, means, covariances = density_estimator.fit(n_components=4)

    # Plot density estimation results here
    # رسم نتایج تخمین چگالی
    plot_density_estimation_results(reduced_dim_data, labels, means, covariances, "Density Estimation of Political Parties")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Plot left and right wing parties here
    pyplot.figure()
    splot = pyplot.subplot()

    # فیلتر کردن احزاب چپ و راست با استفاده از ویژگی lrgen (مقیاس چپ-راست)
    # با فرض اینکه این ویژگی در داده‌های پردازش شده موجود است
    # اگر مقدار lrgen کمتر از صفر باشد، حزب چپ و اگر بیشتر باشد، حزب راست است

    combined = reduced_dim_data.copy()
    combined["lrgen"] = processed_data["lrgen"]
    # تعریف ماسک‌ها برای احزاب چپ و راست
    left_wing_data = combined[combined["lrgen"] < 5].drop(columns=["lrgen"])
    right_wing_data = combined[combined["lrgen"] > 5].drop(columns=["lrgen"])

    print(processed_data["lrgen"].describe())
    print(processed_data["lrgen"].head())
    print(f"Left wing: {len(left_wing_data)}, Right wing: {len(right_wing_data)}")
    # رسم احزاب چپ با رنگ قرمز
    scatter_plot(left_wing_data, color="r", splot=splot, label="Left Wing")

    # رسم احزاب راست با رنگ آبی
    scatter_plot(right_wing_data, color="b", splot=splot, label="Right Wing")

    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")

    # Plot finnish parties here
    # رسم احزاب فنلاندی
    pyplot.figure()
    splot = pyplot.subplot()
    plot_finnish_parties(reduced_dim_data, splot=splot)
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "finnish_parties.png"]))

    print("Analysis Complete")
