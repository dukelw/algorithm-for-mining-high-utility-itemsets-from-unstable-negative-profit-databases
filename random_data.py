import random

data = []

with open('chess.txt', 'r') as file:
    for line in file:
        # Tách các số trong dòng và chuyển đổi chúng thành số nguyên
        numbers = line.split()  # Tách dòng thành danh sách các chuỗi
        
        # Tạo danh sách cho các số trong dòng hiện tại
        row_data = []
        for num in numbers:
            try:
                row_data.append(int(num))  # Chuyển đổi chuỗi thành số nguyên và thêm vào danh sách
            except ValueError:
                # Bỏ qua các giá trị không hợp lệ
                print(f"Đã bỏ qua giá trị không hợp lệ: {num}")
        
        # Thêm danh sách số vào data
        data.append(row_data)


# Tỷ lệ phân chia nhóm
ratios = [0.4, 0.3, 0.3]  # Tỷ lệ cho nhóm dương, âm, hybrid

def create_tid_row(row, ratios, tid):
    # Tạo các item a-z, A-Z cho danh sách items
    items = []
    for i in range(len(row)):
        if i < 26:
            items.append(chr(97 + i))  # Chữ cái thường từ a đến z
        else:
            items.append(chr(65 + (i - 26)))  # Chữ cái hoa từ A đến Z
    
    # Chia số lượng phần tử cho các nhóm dương, âm, hybrid
    positive_count = int(len(row) * ratios[0])
    negative_count = int(len(row) * ratios[1])
    hybrid_count = len(row) - positive_count - negative_count

    positive_profits = row[:positive_count]
    negative_profits = row[positive_count:positive_count + negative_count]
    hybrid_profits = row[positive_count + negative_count:]

    # Áp dụng quy tắc dương, âm, hybrid
    new_profits = []
    quantities = []
    for profit in row:
        quantities.append(1)
        if profit in positive_profits:
            new_profits.append(profit)  # Giữ nguyên giá trị dương
        elif profit in negative_profits:
            new_profits.append(-int(profit))  # Chuyển sang âm
        elif profit in hybrid_profits:
            # Áp dụng cờ ngẫu nhiên để xác định dương hoặc âm
            new_profits.append(profit if random.choice([True, False]) else -profit)

    # Tạo cấu trúc TID
    tid_row = {
        "TID": f"T{tid}",
        "items": items,
        "quantities": quantities,
        "profit": new_profits
    }

    return tid_row

# Khởi tạo danh sách để lưu các TID
dataset = []

# Tạo TID cho từng dòng dữ liệu
for index in range(1, len(data)):
    tid_row = create_tid_row(data[index], ratios, index)
    dataset.append(tid_row)

# Ghi biến dataset vào file data.txt
with open('data.txt', 'w') as outfile:
    for item in dataset:
        outfile.write(f"{item}\n")  # Ghi từng phần tử vào file

# In kết quả để kiểm tra
print(dataset)

