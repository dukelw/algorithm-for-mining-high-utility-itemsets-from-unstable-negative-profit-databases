import random


def process_chess_data(input_file, output_file):
    """
    Hàm xử lý dữ liệu từ file đầu vào, tạo danh sách TID,
    và ghi kết quả vào file đầu ra.

    :param input_file: Đường dẫn đến file đầu vào.
    :param output_file: Đường dẫn đến file đầu ra.
    """
    # Đọc dữ liệu từ file input
    data = []
    with open(input_file, "r") as file:
        for line in file:
            # Tách các số trong dòng và chuyển đổi chúng thành số nguyên
            numbers = line.split()
            row_data = []
            for num in numbers:
                try:
                    row_data.append(int(num))  # Chuyển đổi chuỗi thành số nguyên
                except ValueError:
                    # Bỏ qua các giá trị không hợp lệ
                    print(f"Đã bỏ qua giá trị không hợp lệ: {num}")
            data.append(row_data)

    # Tỷ lệ phân chia nhóm
    full_items = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    total_items = len(full_items)
    ratios = [0.4, 0.3, 0.3]  # Tỷ lệ cho nhóm dương, âm, hybrid
    positive_count = int(total_items * ratios[0])

    # Shuffle items trước khi phân chia
    full_items_original = full_items[:]
    random.shuffle(full_items)

    # Phân chia
    positive_items = full_items[:positive_count]
    negative_items = full_items[positive_count:]

    profits = [random.randint(1, 10) for _ in range(total_items)]

    # Gán giá trị âm cho nhóm negative items
    for idx, item in enumerate(full_items_original):
        if item in negative_items:
            profits[idx] *= -1

    # Hàm tạo TID row
    def create_tid_row(row, tid):
        new_profits = []
        quantities = []
        items = []
        list_index = []

        # Random các chỉ mục
        while len(list_index) < min(len(row), 10):
            item_index = random.randint(0, 9)
            if list_index.count(item_index) == 0:
                list_index.append(item_index)

        # Tạo các giá trị items, profits, quantities
        for i in list_index:
            items.append(full_items_original[i])
            new_profits.append(profits[i])
            quantities.append(random.randint(1, 4))

        return {
            "TID": f"T{tid}",
            "items": items,
            "quantities": quantities,
            "profit": new_profits,
        }

    # Khởi tạo danh sách dataset
    dataset = []
    for index in range(1, min(len(data), 45000)):
        tid_row = create_tid_row(data[index], index)
        dataset.append(tid_row)

    # Ghi kết quả vào file output
    with open(output_file, "w") as outfile:
        outfile.write("[\n")
        outfile.write(",\n".join(map(str, dataset)))
        outfile.write("\n]")

    print(f"Successfully write data to {output_file}")


# Sử dụng hàm
input_file = "retail_real.txt"  # Tên file đầu vào
output_file = "retaildata.txt"  # Tên file đầu ra
process_chess_data(input_file, output_file)
