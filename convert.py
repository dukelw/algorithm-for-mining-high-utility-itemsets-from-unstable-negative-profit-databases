# Đọc file chessdata.txt
with open("chessdata.txt", "r") as file:
    chess_data = file.read()
    chess_data = eval(chess_data)

# Tạo file transactions.txt với định dạng mong muốn
with open("transactions.txt", "w") as transaction_file:
    # Thêm tiêu đề cho file transactions.txt
    transaction_file.write("TID\tItems\tQuantities\tProfit\tUtility\n")

    # Duyệt qua từng dòng trong file chessdata.txt
    for i in range(len(chess_data)):
        transaction = chess_data[i]

        # Lấy dữ liệu từ dictionary
        tid = transaction["TID"]
        items = ",".join(transaction["items"])
        quantities = ",".join(map(str, transaction["quantities"]))
        profits = ",".join(map(str, transaction["profit"]))
        utility = ",".join(
            map(
                str,
                [
                    q * p
                    for q, p in zip(transaction["quantities"], transaction["profit"])
                ],
            )
        )

        # Ghi dữ liệu vào file transactions.txt
        transaction_file.write(f"{tid}\t{items}\t{quantities}\t{profits}\t{utility}\n")

print("Convert successfully!")
