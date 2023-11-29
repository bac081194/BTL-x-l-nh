from flask import Blueprint, render_template, redirect, request, flash
import cv2
import base64
import numpy as np
import os
from PIL import Image

IP = Blueprint("ImageProcessing", __name__)


@IP.route("/negative", methods=["GET", "POST"])
def negative():
    if request.method == "POST":
        # try:
            image = request.files["image"]
            print(image.filename)
            
            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh thành ảnh âm bản
            processedImage = cv2.bitwise_not(originalImage)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/inverted.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/inverted.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        # except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/threshold", methods=["POST", "GET"])
def threshold():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Áp dụng phân ngưỡng (thresholding) với ngưỡng 127 (có thể điều chỉnh)
            # Ảnh nhị phân sẽ có giá trị pixel <= 127 là đen và > 127 là trắng
            _, processedImage = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/threshold.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/threshold.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/logarith", methods=["POST", "GET"])
def logarith():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang định dạng dấu chấm động (float)
            image_float = np.float32(originalImage)

            # Thực hiện biến đổi logarithm
            c = 255 / np.log(1 + np.max(image_float))
            log_image = c * (np.log(image_float + 1))

            # Chuyển đổi về định dạng integer (số nguyên)
            processedImage = np.uint8(log_image)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/logarith.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/logarith.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/gamma", methods=["POST", "GET"])
def gamma():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang định dạng dấu chấm động (float)
            image_float = np.float32(originalImage)
            max_value = np.max(image_float)

            # Hệ số mũ gamma (điều chỉnh độ sáng)
            gamma = 0.5

            # Thực hiện biến đổi hàm mũ
            processedImage = np.uint8(
                np.clip((image_float / max_value) ** gamma * 255.0, 0, 255)
            )

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/gamma.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/gamma.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/balance", methods=["POST", "GET"])
def balance():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang lược đồ xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Cân bằng lược đồ xám
            processedImage = cv2.equalizeHist(gray_image)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/balance.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/balance.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/median", methods=["POST", "GET"])
def median():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Áp dụng bộ lọc trung vị với kích thước cửa sổ 5x5
            processedImage = cv2.medianBlur(originalImage, 5)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/median.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/median.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/weighted_average", methods=["POST", "GET"])
def weighted_average():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Định nghĩa ma trận trọng số
            kernel = (
                np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
            )  # Tính trung bình để chuẩn hóa

            # Áp dụng bộ lọc trung bình có trọng số
            processedImage = cv2.filter2D(originalImage, -1, kernel)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/weighted_average.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/weighted_average.jpg",
                "rb",
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


def knn_filter(image, k):
    kernel = np.ones((k, k), dtype=np.float32) / (
        k * k
    )  # Tạo ma trận kernel với trọng số đồng đều

    # Áp dụng convolution với kernel đã tạo
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image


@IP.route("/knn_average", methods=["POST", "GET"])
def knn_average():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Áp dụng bộ lọc trung bình k giá trị gần nhất với k = 3
            processedImage = knn_filter(originalImage, 3)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/knn_average.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/knn_average.jpg",
                "rb",
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/roberts", methods=["POST", "GET"])
def roberts():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Tạo ma trận kernel cho toán tử Roberts
            roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            roberts_y = np.array([[0, -1], [1, 0]], dtype=np.float32)

            # Áp dụng convolution với kernel của toán tử Roberts
            roberts_x_edges = cv2.filter2D(gray_image, -1, roberts_x)
            roberts_y_edges = cv2.filter2D(gray_image, -1, roberts_y)

            roberts_x_edges = roberts_x_edges.astype(np.float32)
            roberts_y_edges = roberts_y_edges.astype(np.float32)

            # Kết hợp kết quả từ hai hướng (x và y) để tìm biên cuối cùng
            processedImage = cv2.magnitude(roberts_x_edges, roberts_y_edges)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/roberts.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/roberts.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/sobel", methods=["POST", "GET"])
def sobel():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Áp dụng toán tử Sobel theo hướng x và y
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

            # Kết hợp kết quả từ hai hướng (x và y) để tìm biên cuối cùng
            processedImage = cv2.magnitude(sobel_x, sobel_y)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/sobel.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/sobel.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/prewitt", methods=["POST", "GET"])
def prewitt():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển đổi ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Áp dụng toán tử Prewitt theo hướng x và y
            prewitt_kernel_x = cv2.getDerivKernels(1, 0, 3, normalize=True)
            prewitt_kernel_y = cv2.getDerivKernels(0, 1, 3, normalize=True)
            prewitt_x = cv2.filter2D(
                gray_image, cv2.CV_64F, prewitt_kernel_x[0] * prewitt_kernel_x[1]
            )
            prewitt_y = cv2.filter2D(
                gray_image, cv2.CV_64F, prewitt_kernel_y[0] * prewitt_kernel_y[1]
            )

            # Kết hợp kết quả từ hai hướng (x và y) để tìm biên cuối cùng
            processedImage = cv2.magnitude(prewitt_x, prewitt_y)

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/prewitt.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/prewitt.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/laplacian", methods=["POST", "GET"])
def laplacian():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Áp dụng toán tử Laplacian
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

            # Chuyển đổi kết quả thành ảnh có dạng unsigned 8-bit integer
            processedImage = np.uint8(np.absolute(laplacian))

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/laplacian.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/laplacian.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/canny", methods=["POST", "GET"])
def canny():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Áp dụng bộ lọc Gaussian để làm mờ ảnh và làm giảm nhiễu
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

            # Sử dụng phương pháp Canny để phát hiện biên
            processedImage = cv2.Canny(
                blurred_image, 50, 150
            )  # Thay đổi giá trị ngưỡng tùy theo hình ảnh

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/canny.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/canny.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")


# ================================================================================


@IP.route("/otsu", methods=["POST", "GET"])
def otsu():
    if request.method == "POST":
        try:
            image = request.files["image"]
            print(image.filename)

            image.save("images/" + image.filename)
            originalImage = cv2.imread("images/" + image.filename)

            # Chuyển ảnh sang ảnh xám
            gray_image = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)

            # Áp dụng phương pháp Otsu để tìm ngưỡng tối ưu
            _, processedImage = cv2.threshold(
                gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if not os.path.exists("images/" + image.filename.split(".")[0]):
                os.makedirs("images/" + image.filename.split(".")[0])

            cv2.imwrite(
                "images/" + image.filename.split(".")[0] + "/otsu.jpg",
                processedImage,
            )

            with open("images/" + image.filename, "rb") as img:
                originalImage = base64.b64encode(img.read()).decode("utf-8")

            with open(
                "images/" + image.filename.split(".")[0] + "/otsu.jpg", "rb"
            ) as img:
                processedImage = base64.b64encode(img.read()).decode("utf-8")

            return render_template(
                "index.html",
                originalImage=originalImage,
                processedImage=processedImage,
            )
        except:
            flash("Processing error!", "error")

    return render_template("index.html")