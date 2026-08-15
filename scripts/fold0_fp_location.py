import os, glob
import pandas as pd
import SimpleITK as sitk

ROOT="/root/autodl-tmp/nnunet_project"
GT=f"{ROOT}/nnUNet_preprocessed/Dataset003_Liver/gt_segmentations"
RES=f"{ROOT}/nnUNet_results/Dataset003_Liver"

MODELS={
    "A": "nnUNetTrainer",
    "C": "nnUNetTrainer_TransformerBottleneck",
}

rows=[]

for gtfile in sorted(glob.glob(f"{GT}/liver_*.nii.gz")):
    case=os.path.basename(gtfile).replace(".nii.gz","")

    for name,trainer in MODELS.items():
        pred=f"{RES}/{trainer}__nnUNetPlans__3d_fullres/fold_0/validation/{case}.nii.gz"
        if not os.path.exists(pred):
            continue

        ref=sitk.GetArrayFromImage(sitk.ReadImage(gtfile))
        p=sitk.GetArrayFromImage(sitk.ReadImage(pred))

        rows.append({
            "case": case,
            "model": name,
            "outside_fp": int(((p==2)&(ref==0)).sum()),
            "inside_fp": int(((p==2)&(ref==1)).sum()),
        })

df=pd.DataFrame(rows)
df.to_csv("results_csv/fold0_fp_location_AC.csv",index=False)


x=df.pivot(index="case",columns="model",values=["outside_fp","inside_fp"])
x["C_minus_A_outside"]=x["outside_fp"]["C"]-x["outside_fp"]["A"]

print("病例数:",len(x))
print("A总肝外FP:",int(x["outside_fp"]["A"].sum()))
print("C总肝外FP:",int(x["outside_fp"]["C"].sum()))
print("A总肝内FP:",int(x["inside_fp"]["A"].sum()))
print("C总肝内FP:",int(x["inside_fp"]["C"].sum()))
print("C肝外FP增加病例:",int((x["C_minus_A_outside"]>0).sum()))

print("\n肝外FP增加最大的5例:")
print(x.nlargest(5,"C_minus_A_outside")[["C_minus_A_outside"]])
