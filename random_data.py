import random


def generate_tid_data(ratios, number_items, input_file="chess.txt"):
    data = []
    output_file = input_file.split(".")[0] + "data.txt"
    # Danh sách full_items
    full_items = [f"{i}" for i in range(0, number_items)]

    # Tính số lượng items cho từng nhóm
    total_items = len(full_items)
    positive_count = int(total_items * ratios[0])

    # Shuffle items trước khi phân chia
    full_items_original = full_items[:]
    random.shuffle(full_items)

    # Phân chia
    positive_items = full_items[:positive_count]
    negative_items = full_items[positive_count:]

    profits = [random.randint(1, 5) for _ in range(0, total_items)]

    # Gán giá trị âm cho nhóm negative items
    for idx, item in enumerate(full_items_original):
        if item in negative_items:
            profits[idx] *= -1

    # Đọc file đầu vào
    with open(input_file, "r") as file:
        for line in file:
            numbers = line.split()

            row_data = []
            for num in numbers:
                try:
                    row_data.append(int(num))
                except ValueError:
                    print(f"Đã bỏ qua giá trị không hợp lệ: {num}")

            data.append(row_data)

    def create_tid_row(row, tid):
        new_profits = []
        quantities = [random.randint(1, 4) for _ in range(len(row))]
        items = []
        for item in row:
            items.append(full_items_original[item])
            new_profits.append(profits[item])

        return {
            "TID": f"T{tid}",
            "items": items,
            "quantities": quantities,
            "profit": new_profits,
        }

    # Tạo dataset
    dataset = []
    for index in range(1, len(data)):
        tid_row = create_tid_row(data[index], index)
        dataset.append(tid_row)

    # Ghi kết quả ra file
    with open(output_file, "w") as outfile:
        outfile.write("[")
        outfile.write(", ".join(map(str, dataset)))
        outfile.write("]")

    return dataset, positive_items, negative_items


# Ví dụ sử dụng
ratios = [0.3, 0.7]  # Tỷ lệ dương và âm
dataset, positive_items, negative_items = generate_tid_data(ratios, 130, "connect.txt")

print("Positive Items:", positive_items)
print("Negative Items:", negative_items)
print("Dataset:", dataset)
