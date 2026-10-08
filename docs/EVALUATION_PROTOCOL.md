# Protokol Evaluasi Dataset Research

Dokumen ini menjelaskan benchmark untuk fitur deteksi target, task ML, domain, ranking paper, research gap, kualitas data, robustness, dan performa. Evaluasi dijalankan dari modul terpisah; kode alur aplikasi, threshold, prompt, query builder, dan rekomendasi tidak diubah.

## Reproduksibilitas dan privasi

- Seed split dataset: `20261008`. Split dihitung dari SHA-256 `seed:dataset_id`, sehingga isi DEV dan TEST terkunci dan tidak bergantung pada urutan CSV.
- Setiap run mencatat waktu UTC, versi Python/dependensi, mode offline, split, dan seed ke `data/evaluation/results/`.
- Dataset hanya disalin ke `data/evaluation/datasets/`. Hash SHA-256 dicatat sebelum dan sesudah dibaca; sumber data asal tidak ditulis.
- Snapshot OpenAlex/Crossref disimpan lokal beserta tanggal, query, versi httpx, dan hash payload. Gunakan snapshot yang sama untuk perbandingan berulang.
- Berkas label manusia, snapshot, dataset, hasil, serta mapping blind berada di `data/evaluation/`, yang diabaikan Git. Mapping sistem disimpan di `private/` dan tidak diberikan ke penilai.
- Tidak ada angka benchmark, label target, atau label manusia yang ditambahkan sebagai fakta contoh. Fixture sintetik hanya dipakai pada uji kualitas dan performa, serta ditandai sebagai sintetik.

## Menyiapkan dataset ground truth

Isi `evaluation/manifests/dataset_ground_truth.csv` dengan sekurangnya 40 dataset publik. Satu baris untuk satu dataset, dan verifikasi `target_name`, `task_label`, serta `domain_label` dari dokumentasi/sumber dataset. `label_source_url` harus menunjuk sumber yang mendukung label. Nilai split harus hasil `python -c "from evaluation.protocol import locked_split; print(locked_split('ID_DATASET'))"`; gunakan `split_seed=20261008`. Label domain/task yang tidak dapat diverifikasi jangan ditebak; jangan masukkan baris tersebut.

Unduh salinan split DEV secara online atau gunakan cache lokal:

```powershell
python -m evaluation.download_datasets --dev
python -m evaluation.download_datasets --dev --offline
```

Perintah online memerlukan akses OpenML. Cache yang sudah ada dapat digunakan offline. Jangan mengedit salinan CSV setelah hash tercatat; unduh ulang untuk memperbarui secara sah.

## Quick DEV

```powershell
python -m evaluation.run --only all --offline --dev --quick
```

Quick mode menguji utilitas statistik pada contoh kecil, kualitas analyzer terhadap fixture sintetik, input adversarial cepat, dan waktu/RSS pada 1.000 baris dengan lima pengulangan. Modul target/task/domain akan bertanda `TIDAK DIJALANKAN` sampai minimal 40 dataset ground truth dan cache split tersedia. Ranking dan gap menunggu label manusia.

## Snapshot literatur dan label manusia

Ambil snapshot provider menggunakan query benchmark yang sama untuk perbandingan, lalu simpan semua file snapshot untuk run tersebut:

```powershell
python -m evaluation.snapshots --provider openalex --query "isi query benchmark" --limit 20
python -m evaluation.snapshots --provider crossref --query "isi query benchmark" --limit 20
```

Setelah snapshot terkumpul dan pool kandidat aplikasi/baseline diekspor ke JSON array, buat template dengan perintah berikut (opsi `--snapshot` dapat diulang):

```powershell
python -m evaluation.labeling --snapshot data/evaluation/snapshots/SNAPSHOT.json --candidate-pool data/evaluation/candidate_pool.json
```

Setiap `candidate_record` berbentuk `{"query": "query benchmark", "system": "app_ranked", "rank": 1, "paper_id": "DOI atau ID stabil", "title": "judul", "abstract": "abstrak", "publication_year": 2024}`; sediakan baris untuk setiap hasil tiap sistem, termasuk keyword mentah, tanpa re-ranking, sitasi, dan random yang memang tersedia. Kandidat dengan query dan paper ID yang sama dilabel satu kali untuk membentuk union pool. CSV blind dibuat dengan dua baris penilai per paper; isi `relevance_0_2` (0 tidak relevan, 1 sebagian, 2 relevan) dan `rater_id`. Jangan mengubah `blind_id`. Mapping sistem dan urutan tetap privat di `data/evaluation/private/paper_relevance_key.json`.

Untuk gap, simpan kandidat output asli sebagai JSON array, lalu buat template dengan `python -m evaluation.labeling --gaps-json data/evaluation/gaps.json`. Isi setiap pasangan baris penilai: `plausibility_1_5` dan `evidence_sufficiency_1_5`. Nilai menilai plausibilitas dan kecukupan bukti kandidat, bukan kebenaran absolut sebuah gap. Mapping sistem tetap privat.

Untuk menjalankan ablasi, sertakan `score_breakdown` yang benar-benar dihasilkan ranker pada setiap candidate record, misalnya: `{"query":"...","system":"app_ranked","rank":1,"paper_id":"doi:...","title":"...","abstract":"...","publication_year":2024,"score_breakdown":{"semantic":0.8,"dataset_name":0.7,"schema":0.4,"keyword":0.3,"metadata":0.2}}`. Angka di contoh hanya menunjukkan format.

Ranking melaporkan P@10, Recall@10, MRR, nDCG@10 per query/system pada union pool, baseline ranking acak deterministik (seed `20261008`), Cohen's kappa antar penilai, dan Wilcoxon berpasangan jika query cukup. Bila candidate pool menyertakan `score_breakdown` keluaran ranker, runner menghitung ablasi satu komponen per langkah dengan bobot tersisa dinormalisasi untuk evaluasi saja. Ablasi tidak mengubah konfigurasi aplikasi. Gap melaporkan rating, CI bootstrap per sistem, dan Cohen's kappa; tidak dibuat akurasi gap tanpa gold label. Dataset classifier menggunakan accuracy, macro precision/recall/F1, confusion matrix, majority baseline yang hanya dipelajari dari DEV, McNemar exact, serta CI bootstrap.

Evaluasi rekomendasi metode menjalankan cross-validation berpasangan pada dataset DEV dengan preprocessing sklearn yang sama di seluruh estimator. Modul deduplikasi menjalankan fixture sintetik deterministik untuk memastikan DOI/judul/ID yang diketahui menyatu sesuai perilaku kode. Fixture ini bukan estimasi precision/recall pada paper nyata; metrik tersebut memerlukan pasangan paper berlabel manusia. Runner menampilkan `TIDAK DIJALANKAN` jika label paper/gap belum tersedia, skor komponen ranker tidak ada untuk ablasi, atau corpus multi-tahun belum tersedia untuk retrodiksi gap. Status ini bukan hasil nol dan tidak boleh dikutip sebagai hasil evaluasi.

## Evaluasi penuh dan TEST final

Jalankan evaluasi penuh DEV setelah semua dataset dan snapshot/label siap:

```powershell
python -m evaluation.download_datasets --dev --offline
python -m evaluation.run --only all --offline --dev
```

Periksa setiap status `TIDAK DIJALANKAN` atau `GAGAL`, hash, dan sumber label di hasil JSON. Bereskan input/penyebabnya lalu jalankan ulang DEV. Jangan gunakan TEST untuk memilih threshold, prompt, query, atau konfigurasi.

TEST hanya dijalankan setelah laporan DEV dan semua input final ditinjau, label manusia tersedia, serta cache TEST telah disiapkan. Perintah final hanya satu kali:

```powershell
python -m evaluation.download_datasets --test
python -m evaluation.run --only all --offline --test
```

Runner menolak TEST jika manifest, cache, label atau mapping belum lengkap. `data/evaluation/test_run_consumed.json` mencatat pemakaian TEST setelah seluruh modul selesai. Jangan hapus atau mengubah lock tersebut untuk mengulang benchmark pada split yang sama. Untuk eksperimen setelah melihat TEST, siapkan benchmark TEST baru yang belum pernah dibuka.

## Batas interpretasi

- Hasil benchmark adalah bukti pada dataset, tanggal snapshot, dan label yang dicatat; bukan jaminan performa untuk semua penggunaan.
- Ranking dinilai pada pool kandidat snapshot yang diambil, bukan seluruh publikasi yang mungkin ada.
- Label domain/task dan relevansi dipengaruhi cakupan dokumentasi serta kesepakatan penilai; tampilkan ukuran sampel, confusion matrix, agreement, dan interval ketidakpastian.
- Performance baseline pandas mengukur operasi pembanding yang ditulis di runner, bukan implementasi produk alternatif dengan keluaran identik. RSS diambil dengan sampling 10 ms sehingga alokasi sangat singkat dapat tidak tertangkap.
- Setiap error, crash, dan modul yang tidak dijalankan harus tetap dicatat; jangan menghapus hasil buruk dari laporan.
