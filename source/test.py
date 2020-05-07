from source.yarpp import *

if __name__ == "__main__":
    img_path_list = ["/Users/wilsonzhang/Documents/PROJECTS/Personal/YARPP/sample_data/apple-logo.jpg",
                     "/Users/wilsonzhang/Documents/PROJECTS/Personal/YARPP/sample_data/temp_A.png"]
    for i, img_path in enumerate(img_path_list):
        print("Processing image {}".format(i))
        yarpp = YARPP(img_path)
        yarpp.helper_plot(type="circle", sample_size=300, annotate=False, random_color=True)
