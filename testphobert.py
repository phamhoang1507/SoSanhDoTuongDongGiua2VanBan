


import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

# Tải pre-trained model và tokenizer
model_name = "vinai/phobert-base-v2"
phobert = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Hàm để chuyển đoạn văn thành vector
def get_vector(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = phobert(input_ids)
    vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return vector

# Đoạn văn bản 1
text1 = "Con Cá Chuối Này To Nấu Chắc Phải 5 Người Ăn Không Hết"

# Đoạn văn bản 2
text2 = "Tôi Cá Rằng Ông Này Mai Nghỉ Học"

# Tạo vector cho mỗi đoạn văn bản
vector1 = get_vector(text1)
vector2 = get_vector(text2)

# Tính cosine similarity giữa hai vector
similarity = cosine_similarity([vector1], [vector2])[0][0]

# In độ tương đồng
print("Vector 1 : ",vector1)
print("")
print("Vector 2 : ",vector2)
print(f"Độ tương đồng giữa '{text1}' và '{text2}': {similarity}")
if(similarity<0.78):
    print("Văn Bản Có Độ Tương Đồng Thấp")
elif(similarity>0.78 and similarity<0.9):
    print("Văn Bản Có Độ Tương Đồng Cao")
else:
    print("Văn Bản Có Độ Tương Đồng Rất Cao")


