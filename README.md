# Sistem Rekomendasi Pemain Sepak Bola

Indonesian Player Recommendation System (IPRS) adalah aplikasi berbasis web yang dirancang untuk membantu proses rekomendasi pemain sepak bola berdasarkan kemiripan karakteristik performa. Sistem ini memanfaatkan metode *unsupervised learning* untuk melakukan clustering pemain dan perhitungan kemiripan statistik sebagai dasar rekomendasi. Aplikasi ini menggunakan **Streamlit** sebagai antarmuka pengguna, **Django** sebagai backend, dan **PostgreSQL** sebagai basis data utama.

## Fitur Utama
- Unggah dataset statistik pemain sepak bola
- Validasi dan penyimpanan data ke database PostgreSQL
- Pemilihan data berdasarkan musim kompetisi
- Proses clustering pemain berdasarkan kesamaan performa
- Pemilihan pemain acuan
- Rekomendasi pemain berdasarkan tingkat kemiripan statistik
- Visualisasi hasil clustering dan perbandingan statistik pemain

## Teknologi yang Digunakan
- **Python**
- **Streamlit** – Antarmuka pengguna
- **Django** – Backend dan manajemen database
- **PostgreSQL** – Database
- **Pandas & NumPy** – Pengolahan data
- **Scikit-learn** – Clustering dan similarity
- **Altair / Matplotlib / Seaborn** – Visualisasi data

## Arsitektur Sistem
Sistem menggunakan arsitektur terpisah antara antarmuka dan backend. Streamlit berfungsi sebagai antarmuka pengguna sekaligus pengendali proses analisis dan visualisasi, sedangkan Django bertanggung jawab atas pengelolaan database, validasi data, serta penyimpanan dataset. PostgreSQL digunakan sebagai basis data utama untuk menyimpan data pemain, musim, dan hasil analisis.

## Alur Penggunaan
1. Pengguna membuka aplikasi Streamlit.
2. Pengguna mengunggah dataset pemain atau menggunakan data yang sudah tersedia.
3. Sistem memvalidasi dan menyimpan data ke database PostgreSQL melalui Django.
4. Pengguna memilih musim kompetisi yang akan dianalisis.
5. Sistem menjalankan proses clustering pemain.
6. Pengguna memilih pemain acuan.
7. Sistem menampilkan rekomendasi pemain berdasarkan kemiripan performa.
8. Pengguna melihat visualisasi dan hasil rekomendasi pemain.

## Instalasi dan Menjalankan Aplikasi

### 1. Clone Repository
```bash
git clone https://github.com/username/iprs.git
cd iprs
```

### 2. Instalasi Dependency
```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Database
Pengaturan Database dilakukan langsung pada file settings.py di project Django

### 4. Migrasi Database
```bash
python manage.py migrate
```

### 5. Jalankan aplikasi
```bash
streamlit run streamlit_app.py
```
