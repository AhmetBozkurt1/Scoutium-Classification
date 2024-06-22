<img width="700" alt="Ekran Resmi 2024-06-23 01 13 17" src="https://github.com/AhmetBozkurt1/Scoutium-Classification/assets/120393650/68c72eec-d8fe-48ec-ab70-2342cd17e36e">

# TALENT HUNT CLASSIFICATION WITH MACHINE LEARNING

### İŞ PROBLEMİ
Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

### VERİ SETİ HİKAYESİ
Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

### DEĞİŞKENLER
- #### scoutium_attributes.csv
    - **task_response_id:** Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
    - **match_id:** İlgili maçın id'si
    - **evaluator_id:** Değerlendiricinin(scout'un) id'si
    - **player_id:** İlgili oyuncunun id'si
    - **position_id:** İlgili oyuncunun o maçta oynadığı pozisyonun id’si
        - 1: Kaleci 
        - 2: Stoper 
        - 3: Sağ bek 
        - 4: Sol bek
        - 5: Defansif orta saha
        - 6: Merkez orta saha
        - 7: Sağ kanat
        - 8: Sol kanat
        - 9: Ofansif orta saha
        - 10: Forvet
    - **analysis_id:** Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
    - **attribute_id:** Oyuncuların değerlendirildiği her bir özelliğin id'si
    - **attribute_value:** Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)

- #### scoutium_potential_labels.csv
    - **task_response_id:** Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
    - **match_id:** İlgili maçın id'si
    - **evaluator_id:** Değerlendiricinin(scout'un) id'si
    - **player_id:** İlgili oyuncunun id'si
    - **potential_label:** Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)
 
### MODEL OLUŞTURMA
- Veri seti keşfedilir ve özelliklerin analizi yapılır.
- Eksik veriler ve aykırı değerler işlenir.
- Özellik mühendisliği adımlarıyla yeni özellikler türetilir.
- Kategorik değişkenler sayısal formata dönüştürülür.
- Model seçimi yapılır ve hiperparametre optimizasyonu gerçekleştirilir.
- En iyi modelin performansı değerlendirilir.


### Gereksinimler
☞ Bu proje çalıştırılmak için aşağıdaki kütüphanelerin yüklü olması gerekmektedir:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- lightgbm
- xgboost
- catboost

### Kurulum
☞ Projeyi yerel makinenizde çalıştırmak için şu adımları izleyebilirsiniz:
- GitHub'dan projeyi klonlayın.
- Projeyi içeren dizine gidin ve terminalde `conda env create -f environment.yaml` komutunu çalıştırarak gerekli bağımlılıkları yükleyin.
- Derleyicinizi `conda` ortamına göre ayarlayın.
- Projeyi bir Python IDE'sinde veya Jupyter Notebook'ta açın.
