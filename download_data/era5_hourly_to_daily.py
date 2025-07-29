
import os
import xarray as xr

def calculate_daily_averages(hourly_ds, variables):
    """Convert hourly data to daily averages."""
    return xr.Dataset({var: hourly_ds[var].resample(valid_time='1D').mean() for var in variables})

def convert_hourly_to_daily(input_dir, output_dir, variables, years, prefix="era5"):
    """Process hourly files into daily NetCDF datasets."""
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        file_path = os.path.join(input_dir, f"{prefix}_{variables[0]}_hourly_{year}.nc")
        ds = xr.open_dataset(file_path)

        daily_ds = calculate_daily_averages(ds, variables)
        output_path = os.path.join(output_dir, f"{prefix}_{variables[0]}_daily_{year}.nc")
        daily_ds.to_netcdf(output_path)

        print(f"Saved daily averages: {output_path}")

if __name__ == "__main__":
    years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2021']
    variables = ['slhf', 'snsr', 'sntr', 'sshf']
    input_dir = 'data/era5/raw'
    output_dir = 'data/era5/daily'
    convert_hourly_to_daily(input_dir, output_dir, variables, years, prefix="era5")
