import kagglehub

# Download latest version
path = kagglehub.competition_download('filament-segmentation-2026')

print("Path to competition files:", path)