import cv2
import numpy as np
import os

# Hàm XOR giữa hai ảnh nhị phân (sử dụng từng kênh màu)
def xor_image(image1, image2):
    return cv2.bitwise_xor(image1, image2)

# Chuyển ảnh thành ảnh nhị phân
def binarize_image(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    return binary_image

# Tạo ma trận ngẫu nhiên M_even và M_odd
def generate_random_matrices(h, w):
    M_even = np.random.randint(0, 2, size=(h, w), dtype=np.uint8) * 255
    M_odd = np.random.randint(0, 2, size=(h, w), dtype=np.uint8) * 255
    return M_even, M_odd

# Tạo các chia sẻ từ ảnh bí mật và ảnh phủ sử dụng phương pháp XOR cải tiến
def generate_shares(secret_image, cover_image, n, k):
    secret_image = binarize_image(secret_image)  # Chuyển ảnh bí mật thành nhị phân
    h, w = secret_image.shape
    shares = []

    # Tạo ma trận M_even và M_odd
    M_even, M_odd = generate_random_matrices(h, w)

    # Tách ảnh phủ thành ba kênh (Red, Green, Blue)
    cover_image_bgr = cv2.split(cover_image)  # Tách các kênh của ảnh màu

    # Tạo (n-1) chia sẻ ngẫu nhiên và lấy hàng ngẫu nhiên từ M_even và M_odd
    random_shares = []
    for i in range(n - 1):
        r = np.random.randint(0, 2)  # Chọn ngẫu nhiên r trong {0, 1}
        if r == 0:
            D = M_even
        else:
            D = M_odd
        share = np.copy(secret_image)
        for i in range(h):
            for j in range(w):
                # Thực hiện XOR giữa các giá trị pixel của ảnh bí mật và ảnh phủ
                share[i, j] = D[i, j] ^ secret_image[i, j]  # XOR với ảnh phủ
        random_shares.append(share)

    # Tạo chia sẻ cuối cùng với phép XOR của tất cả các chia sẻ ngẫu nhiên
    final_share = secret_image.copy()
    for rand_share in random_shares:
        final_share = xor_image(final_share, rand_share)
    random_shares.append(final_share)

    # Pha trộn các chia sẻ với ảnh phủ màu và thêm vào danh sách chia sẻ
    for share in random_shares:
        # Pha trộn từng kênh với chia sẻ
        blended_share_bgr = []
        for i in range(3):  # Xử lý 3 kênh (Blue, Green, Red)
            blended_channel = blend_with_cover(share, cover_image_bgr[i])
            blended_share_bgr.append(blended_channel)
        
        # Kết hợp lại thành ảnh màu
        blended_share = cv2.merge(blended_share_bgr)
        shares.append(blended_share)

    return shares

# Pha trộn chia sẻ với từng kênh màu của ảnh phủ
def blend_with_cover(binary_share, cover_channel):
    return cv2.addWeighted(cover_channel, 0.5, binary_share, 0.5, 0)

# Tái tạo ảnh bí mật từ các chia sẻ
def reconstruct_secret(shares, k):
    if len(shares) < k:
        print(f"Lỗi: Cần ít nhất {k} chia sẻ để tái tạo ảnh bí mật.")
        return None
    
    # Chuyển các chia sẻ thành ảnh nhị phân
    binary_shares = [binarize_image(share) for share in shares]
    reconstructed = binary_shares[0]
    for i in range(1, k):
        reconstructed = xor_image(reconstructed, binary_shares[i])
    return reconstructed

# Lưu ảnh chia sẻ vào thư mục
def save_images(images, output_dir, prefix="share"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, image in enumerate(images):
        filename = f"{output_dir}/{prefix}_{i + 1}.png"
        cv2.imwrite(filename, image)
        print(f"Đã lưu chia sẻ: {filename}")

if __name__ == "__main__":
    # Đọc ảnh gốc và ảnh phủ
    secret_image_path = "C:/Users/Loc/Pictures/out/goc.jpg"
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_GRAYSCALE)
    if secret_image is None:
        print("Lỗi: Không tìm thấy ảnh bí mật.")
        exit()

    cover_image_path = "C:/Users/Loc/Pictures/out/phu.jpg"
    cover_image = cv2.imread(cover_image_path)  # Đọc ảnh màu (BGR)
    if cover_image is None:
        print("Lỗi: Không tìm thấy ảnh phủ.")
        exit()

    # Số lượng chia sẻ (n) và số chia sẻ cần thiết để tái tạo ảnh (k)
    n = 90 # Số chia sẻ muốn tạo
    k = 90 # Số chia sẻ cần thiết để tái tạo ảnh

    # Tạo chia sẻ
    shares = generate_shares(secret_image, cover_image, n, k)

    # Lưu các chia sẻ
    save_images(shares, output_dir="output_shares", prefix="share")

    # Tái tạo ảnh bí mật từ các chia sẻ (chỉ dùng k chia sẻ)
    reconstructed_secret = reconstruct_secret(shares[:k], k)

    if reconstructed_secret is not None:
        # Lưu và hiển thị ảnh bí mật đã tái tạo
        cv2.imwrite("reconstructed_secret.png", reconstructed_secret)
        print("Ảnh bí mật đã tái tạo được lưu dưới tên 'reconstructed_secret.png'")

        # Hiển thị ảnh tái tạo
        cv2.imshow("Reconstructed Secret", reconstructed_secret)
        cv2.waitKey(0)
        cv2.destroyAllWindows()