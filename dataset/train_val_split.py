import splitfolders

input_folder = "D:\RecipePal\DATASETS"

output = "dataset\data"

splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .2))
