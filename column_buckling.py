import numpy as np
from scipy.optimize import bisect

def find_critical_load(L, E, A, r, c, e, sigma_allow):
    """
    L: אורך במ"מ
    E: מודול אלסטיות ב-MPa
    A: שטח חתך בממ"ר
    r: רדיוס אינרציה במ"מ
    c: מרחק לסיב קיצוני במ"מ
    e: אקסצנטריות במ"מ
    sigma_allow: מאמץ מותר ב-MPa
    
    Return: העומס P בניוטון (float)
    """
    
    # חישוב עומס הקריסה של אוילר שמהווה את החסם העליון התיאורטי
    # P_euler = (pi^2 * E * I) / L^2
    # נציב I = A * r^2:
    P_euler = (np.pi**2 * E * A) / (L / r)**2
    
    # פונקציית עזר למציאת השורש: f(P) = sigma_max(P) - sigma_allow
    def f(P):
        # טיפול במקרה קצה בו העומס הוא 0 כדי למנוע חלוקה באפס או שגיאות חישוב
        if P == 0:
            return -sigma_allow
            
        # חישוב הביטוי שבתוך ה-secant (ברדיאנים)
        theta = (L / (2 * r)) * np.sqrt(P / (E * A))
        
        # חישוב הפונקציה secant כ- 1 חלקי קוסינוס
        sec_val = 1.0 / np.cos(theta)
        
        # הצבה בנוסחת הסקנט לחישוב המאמץ המקסימלי
        sigma_max = (P / A) * (1 + (e * c / r**2) * sec_val)
        
        # אנו רוצים שההפרש בין המאמץ המחושב למותר יהיה אפס
        return sigma_max - sigma_allow

    # שימוש בשיטת החצייה (Bisection) למציאת העומס P
    # אנו מחפשים את השורש בקטע שבין 0 ל- 99.99% מעומס אוילר 
    # (כדי לא להגיע בדיוק לאסימפטוטה של פונקציית הסקנט שם היא שואפת לאינסוף)
    P_critical = bisect(f, 0, 0.9999 * P_euler)
    
    return float(P_critical)
