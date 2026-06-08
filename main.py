import cv2
from insightface.app import FaceAnalysis
from src.region_masking import mask_region

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0)

img = cv2.imread("data/raw_images/test.jpg")

faces = app.get(img)

print("Faces detected:", len(faces))

if len(faces) == 0:
    exit()

face = faces[0]

x1, y1, x2, y2 = map(int, face.bbox)
face_crop = img[y1:y2, x1:x2]

regions = ["eyes", "nose", "mouth", "upper", "lower"]

for r in regions:
    masked = mask_region(face_crop, r)

    path = f"data/masked_images/{r}.jpg"
    cv2.imwrite(path, masked)

    print("Saved:", path)

from src.embedding import cosine_similarity

# full face embedding
full_embedding = face.embedding

results = []

regions = ["eyes", "nose", "mouth", "upper", "lower"]

for r in regions:
    img_masked = cv2.imread(f"data/masked_images/{r}.jpg")
    faces_masked = app.get(img_masked)

    if len(faces_masked) == 0:
        continue

    emb = faces_masked[0].embedding

    sim = cosine_similarity(full_embedding, emb)

    results.append((r, sim))
    print(f"{r} similarity: {sim:.4f}")

print("\nFINAL RESULTS:")
for r, s in results:
    print(r, "->", s)