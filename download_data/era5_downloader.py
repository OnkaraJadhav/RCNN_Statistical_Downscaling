
import cdsapi
import os

def download_era5_data(variables, years, area, output_dir, prefix="era5"):
    """
    Downloads ERA5 hourly data from Copernicus CDS.
    """
    c = cdsapi.Client()
    os.makedirs(output_dir, exist_ok=True)

    for year in years:
        output_file = os.path.join(output_dir, f"{prefix}_{variables[0]}_hourly_{year}.nc")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': variables,
                'year': year,
                'month': [f"{i:02d}" for i in range(1, 13)],
                'day': [f"{i:02d}" for i in range(1, 32)],
                'time': [f"{i:02d}:00" for i in range(24)],
                'area': area,
            },
            output_file
        )
        print(f"Download complete: {output_file}")

if __name__ == "__main__":
    variables = ['surface_net_thermal_radiation']
    years = ['2021']
    area = [-22.5763, 108.511, -34.3265, 116.284]
    output_dir = 'data/era5/raw'
    download_era5_data(variables, years, area, output_dir, prefix="era5_sntr")
