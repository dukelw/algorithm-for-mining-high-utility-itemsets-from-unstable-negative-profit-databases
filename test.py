# Đọc dữ liệu từ file table.txt và lưu vào biến dataset
with open("table.txt", "r") as file:
    data = file.read()  # Đọc toàn bộ nội dung file

# Chuyển nội dung chuỗi thành danh sách Python
dataset = eval(data)

# In dataset để kiểm tra
print(dataset)
