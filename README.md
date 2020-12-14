### 2020 NUS-MIT Datathon 
### Team 47

## Title: Predicting the Requirement of Ventilator with Chest X-ray and Vital signs 24 hours eariler

** This repository contains only  the Chest X-ray part of the project. To know futher about the project please contact other contributors mentioned below**

Deciding when to put a patient on a mechanical ventilator(MV) is crucial for the survival of ICU patients. Unfortunately, with the recent COVID-19 pandemic, limited ICU beds and shortage of MV have become a bottleneck in the provision of care to the critically ill patients. If the patient fails to get ventilated at the right time, not only the risk of death is increased, but also the infection is deepened socially, resulting in greater economic and human losses.

While the criteria for applying a MV has been made empirically by trained-clinicians, this will not be enough in the COVID-19 pandemic due to the shortage of skillful ICU doctors and MV. This problem was already reported in many hospitals of the countries suffering from COVID-19 outbreak such as the US and Europe.

Therefore, to solve this problem our team propose the DL model by using chest X-ray(CXR) and EHR data including vital signs and lab tests to predict the requirements of MV 24hrs earlier. While many studies in the past have already tried using MIMIC-3 or other EHR records to build AI based models for the prediction of ventilator requirements earlier, to the best of our knowledge we are the first team to incorporate CXR into the modeling.

In this Datathon, we develop the DL models to predict the requirements of MV under 3 different scenarios : ** 1) using only the Chest X-ray images from the CXR dataset ** ; We will then compare the performance of the 3 individual models. 2) using only the EHR records from MIMIC-4 dataset; and 3) combining the two datasets. 

Our model can help young clinicians to agilely respond to life-threatening ICU emergency patients and also allocates MV more effectively with the prediction of ventilator requirements earlier.

## Requirements
* **Datasets** : [MIMIC-IV Datasets](https://mimic-iv.mit.edu/), [MIMIC-CXR-JPEG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).
* Packages : refer to requirements.txt
* In this repository, we do not share the cohort file. To run our code, you first have to extract your own desired cohort. Your cohort file should have split: train | test , mv_flag: MV | non MV, subject_id, hadm_id, stay_id, hosp_admissiontime, cu_admissiontime, mvstarttime_or_icu_dischargetime, dicom_id: jpeg file name, study_id, viewposition : AP|PA, cxrstudytime : time when the CXR was taken, studydate, studytime. 

## TO RUN

simply type :
> python main.py


## Contributors

* Young Seok  Jeon / Ph.D. Student  / Deep learning  / youngseokjeon74@gmail.com
* Jung Hwa Lee / MD.PhD. / Clinical associate Professor / Study concept&design,Clinical advice /  jhlee116@ewha.ac.kr
* Seungmin Baik/ MD/ Clinical associate Professor / study design, medical advice / illaillailla@naver.com
* Seung Won Lee / MD.PhD./ Asst. Professor / Statistical modeling / swlsejong@sejong.ac.kr
* Seojeong Shin / Ph.D. Student / Pathology Image Deep Learning / lucid90sj@gmail.com
* Doyeop Kim / Integrated-PhD Student / SQL Expert / hidoyebi@ajou.ac.kr
* Chioh Song / Linewalks / data scientist / ghsehr1@gmail.com 
* Cinyoung Hur / Linewalks / SQL / cinyoung.hur@gmail.com
* Geun U Park  / Linewalks / SQL  / geunu@linewalks.com
* Haneol  Lee / Biomedical Engineering in Yonsei Univ, KAIST  / studymode@yonsei.ac.kr
