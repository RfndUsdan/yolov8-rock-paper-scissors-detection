import cv2
from ultralytics import YOLO

# 1. Load model 
model = YOLO("best_model_rock_paper_scissors.pt")

# 2. Nyalakan kamera laptop 
cap = cv2.VideoCapture(0)

print("Kamera aktif! Arahkan tanganmu ke kamera.")
print("Tekan tombol 'q' di keyboard untuk keluar dari video.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca kamera.")
        break

    # 3. Lakukan prediksi secara real-time 
    results = model(frame, stream=True, verbose=False)

    # 4. Gambar kotak deteksi 
    for r in results:
        frame = r.plot()

    cv2.imshow("YOLOv8 - Rock Paper Scissors Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()