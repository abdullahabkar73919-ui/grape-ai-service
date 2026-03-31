
import cv2
import os
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from ultralytics import YOLO

DETECTOR_MODEL_PATH = "model_AI/detector_best.pt"
DISEASE_MODEL_PATH = "model_AI/disease_best.pt"   




# عتبات الثقة (Confidence Thresholds)
CONF_THRES_DETECTOR = 0.7
CONF_THRES_DISEASE = 0.4
CONF_THRES_HEALTHY = 0.6

# القواميس اللغوية للترجمة والتقرير
DETECTOR_TRANSLATION = {
    "vines_leaf": "الورقة",
}

DISEASE_TRANSLATION = {
    "powdery_mildew": "البياض الدقيقي",
    "downy_mildew": "البياض الزغبي",
    "black_rot": "العفن الأسود",
    "healthy_leaf": "سليمة"
}

# تعريف الفئات (تأكد من مطابقتها لملف data.yaml الخاص بك)
DETECTOR_CLASSES = {0: "vines_leaf",  2: "other_leaf"}
DISEASE_CLASSES = {0: "black_rot", 1: "downy_mildew", 2: "powdery_mildew"}

# ==========================================================
# 2. تهيئة التطبيق وتحميل النماذج
# ==========================================================
app = FastAPI(title="Grape Guardian AI System")

device = 0 if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# تحميل النماذج إلى الذاكرة مع تحقق من وجود الملفات ومعالجات خطأ أوضح
if not os.path.exists(DETECTOR_MODEL_PATH):
    raise FileNotFoundError(f"Detector model not found at {DETECTOR_MODEL_PATH}")
if not os.path.exists(DISEASE_MODEL_PATH):
    raise FileNotFoundError(f"Disease model not found at {DISEASE_MODEL_PATH}")

try:
    detector_model = YOLO(DETECTOR_MODEL_PATH).to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load detector model {DETECTOR_MODEL_PATH}: {e}")

try:
    disease_model = YOLO(DISEASE_MODEL_PATH).to(device)
except Exception as e:
    raise RuntimeError(f"Failed to load disease model {DISEASE_MODEL_PATH}: {e}")

# ==========================================================
# 3. الوظائف المساعدة (Helper Functions)
# ==========================================================
def valid_mask(mask, img_shape):
    """التحقق من أن مساحة الماسك منطقية (ليست ضجيجاً وليست كامل الصورة)"""
    h, w = img_shape[:2]
    area_ratio = np.sum(mask) / (h * w)
    return 0.0001 <= area_ratio <= 0.95

def is_grape_present(img):
    """المرحلة الأولى: التأكد من وجود عنب وتحديد نوع الجزء المصور"""
    results = detector_model.predict(source=img, conf=CONF_THRES_DETECTOR, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [0]:  # 0 لورقة، 1 لعنقود
                return True, DETECTOR_CLASSES[cls_id]
    return False, None

def diagnose_disease(img):
    """المرحلة الثانية: تشخيص المرض.

    مطابقة لمنطق `predict_from_image` القديم: نستخدم عتبة `CONF_THRES_DISEASE` عند الاستدعاء
    ونعيد حالات مشابهة (no_leaf_detected, disease_detected, healthy, uncertain).
    """
    results = disease_model.predict(source=img, conf=CONF_THRES_DISEASE, device=device, retina_masks=True, verbose=False)

    disease_conf = {}
    healthy_conf = None
    found_target = False

    for r in results:
        if r.masks is None:
            continue

        for i, mask in enumerate(r.masks.data):
            cls_id = int(r.boxes.cls[i])
            conf = float(r.boxes.conf[i])

            if cls_id not in DISEASE_CLASSES:
                continue

            binary_mask = mask.cpu().numpy() > 0.5
            if not valid_mask(binary_mask, img.shape):
                continue

            found_target = True

            if cls_id in [0, 1, 2] and conf >= CONF_THRES_DISEASE:
                disease_conf[DISEASE_CLASSES[cls_id]] = max(disease_conf.get(DISEASE_CLASSES[cls_id], 0), conf)

            elif cls_id == 3 and conf >= CONF_THRES_HEALTHY:
                healthy_conf = max(healthy_conf or 0, conf)

    if not found_target:
        return {"status": "no_leaf_detected"}

    if disease_conf:
        best_disease = max(disease_conf, key=disease_conf.get)
        return {
            "status": "disease_detected",
            "class": best_disease,
            "confidence": round(disease_conf[best_disease] * 100, 2)
        }

    # Do not return 'healthy' status; treat healthy as 'uncertain' per preference
    if healthy_conf:
        return {"status": "uncertain"}

    return {"status": "uncertain"}

# ==========================================================
# 4. نقطة النهاية (API Endpoint)
# ==========================================================
@app.post("/diagnose")
async def main_diagnose_process(file: UploadFile = File(...)):
    try:
        # قراءة وتحويل الصورة
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="خطأ في تنسيق الصورة")

        # --- المرحلة الأولى: الكشف ---
        found_grape, detected_type = is_grape_present(img)
        
        if not found_grape:
            return {
                "status": "not_grape",
                "message": "لم يتم التعرف على العنب. يرجى تصوير ورقة بوضوح."
            }

        # --- المرحلة الثانية: التشخيص ---
        diagnosis = diagnose_disease(img)
        
        # --- المرحلة الثالثة: صياغة التقرير النهائي باللغة العربية ---
      # --- المرحلة الثالثة: صياغة التقرير النهائي باللغة العربية ---
        obj_ar = DETECTOR_TRANSLATION.get(detected_type, "الجزء المكتشف")
        
        # تحديد المتغير الذي سيحمل اسم المرض
        final_disease_class = diagnosis.get("class")

        if diagnosis["status"] == "disease_detected":
            # هذه الدالة ستقوم بتحويل أي مرض من الثلاثة إلى اسمه العربي مباشرة
            disease_ar = DISEASE_TRANSLATION.get(diagnosis["class"], diagnosis["class"])
            
            # تعيين الاسم العربي ليكون هو القيمة التي سيتم إرسالها
            final_disease_class = disease_ar 
            
            final_report = f"تم اكتشاف مرض {disease_ar} في {obj_ar}."
        else:
            final_report = f"تم العثور على {obj_ar} ولكن الحالة الصحية غير واضحة."

        # تجميع الرد النهائي
        return {
            "status": diagnosis["status"],
            "detected_object": obj_ar,
            "class": final_disease_class, # سيرسل (البياض الزغبي أو البياض الدقيقي أو العفن الأسود)
            "confidence": diagnosis.get("confidence"),
            "damage_percent": diagnosis.get("damage_percent"),
            "final_report": final_report
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)