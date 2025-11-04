import pandas as pd
from django.core.exceptions import ValidationError
import datetime as dt
from players.services import TEMPLATE_DATA, build_template_file, delete_dataset, get_list_of_season, post_dataset


def get_unggah_dataset_page(st):
    st.header("Template Dataset")

    # DOWNLOAD FILE TEMPLATE DATASET
    df = pd.DataFrame(TEMPLATE_DATA)
    st.dataframe(df, width=760)
    st.download_button(
        "Download Template",
        data=build_template_file(df),
        file_name=f"template_dataset_{dt.datetime.now():%Y%m%d}.xlsx",
        # CONTENT TYPE XLSX
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # UPLOAD FILE DATASET
    st.markdown("---")
    st.header("Unggah Dataset")
    st.session_state.setdefault("uploader_key", 0)

    with st.form("upload_form", width=760):
        league_name = st.text_input("Nama Liga", value="Liga 1 Indonesia")
        season = st.text_input("Musim", placeholder=f"misal 2024/2025", value="2024/2025")
        file = st.file_uploader("Unggah file dataset", type="xlsx", key=f"upload_{st.session_state['uploader_key']}")
        submitted = st.form_submit_button("Simpan")

    if submitted:
        try:
            # VALIDASI DATA
            if not league_name:
                st.error("Isi nama liga terlebih dahulu.")
            if not season:
                st.error("Isi musim terlebih dahulu.")
            if not file:
                st.error("Unggah file dataset terlebih dahulu.")
            else:
                dataset_file = pd.read_excel(file)
                post_dataset(league_name, season, dataset_file)
                st.success(f"Sukses menyimpan dataset: {league_name} – {season}.")
                st.session_state["uploader_key"] += 1
                st.rerun()
        except KeyError as ke:
            st.error(str(ke))
        except ValueError as ve:
            st.error(str(ve))
        except ValidationError as validation_error:
            st.error(str(validation_error))
        except Exception as e:
            st.error(f"Gagal memproses file: {e}")

    st.markdown("---")
    
    # MENAMPILKAN DATA LIGA YANG SUDAH DIUNGGAH
    st.header("Data Yang Sudah Diunggah")
    seasons = get_list_of_season()

    # YANG DITAMPILKAN JIKA BELOM ADA DATA YANG DISIMPAN
    if not seasons:
        st.info("Belum ada data yang tersimpan")
    else:
        # DATA MUSIM YANG SUDAH DIUNGGAH
        col_head1, col_head2, col_head3, col_head4, col_head5 = st.columns([1, 1, 1, 1, 1])
        col_head1.write("**Liga**")
        col_head2.write("**Musim**")
        col_head3.write("**Jumlah Pemain**")
        col_head4.write("**Diunggah**")

        for ds in seasons:
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            col1.write(ds["league_name"])
            col2.write(ds["season"])
            col3.write(ds["player_count"])
            col4.write(pd.to_datetime(ds["uploaded_at"]).strftime("%d-%m-%Y"))
            
            # BUTTON HAPUS DATA LIGA
            if col5.button("Hapus", key=f"del_{ds['id']}"):
                ok = delete_dataset(ds["id"])
                if ok:
                    st.success(f"Data {ds['league_name']} musim ({ds['season']}) berhasil dihapus.")
                    st.rerun()
                else:
                    st.error("Gagal menghapus data liga.")
                    
    return st