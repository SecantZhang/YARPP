from source.yarpp import *

if __name__ == "__main__":
    img_path = "/Users/wilsonzhang/Documents/PROJECTS/Personal/YARPP/sample_data/apple-logo.jpg"
    yarpp = YARPP(img_path)
    yarpp.helper_plot(type="circle", sample_size=100, annotate=True)
