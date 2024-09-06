import streamlit as st

from model import KNK


@st.cache_data
def get_data(f, queries):
    st.write("Загрузка данных и инициализация методов...")
    knk_creator = KNK(queries_list=queries)
    st.write("Модель готова к работе.")
    st.write("Создание таблицы КНК...")
    knk_df, new_df, full_char_list = knk_creator.create(f)
    st.write("Готово!")
    return knk_df, new_df, full_char_list


queries = ["навыки", "знания", "компетенции", "должности", "Soft Skills"]

st.title("Анализ профиля компетенций 💼")
file = st.file_uploader("Загрузить должностную инструкцию", type="docx")

if file:
    with st.status("Обработка файла..."):
        knk_df, new_df, full_char_list = get_data(file, queries)
    if knk_df is not None:
        st.header("Должность")
        df_position = knk_df[knk_df.columns[:3]].iloc[[0]]
        st.dataframe(df_position)

        soft_skills = [
            item for item in knk_df["Софт-скиллы"].dropna().values if len(item) > 0
        ]
        if len(soft_skills) != 0:
            st.header("Софт-скиллы")
            s = ""
            for skill in soft_skills:
                s += "- " + skill + "\n"
            st.markdown(s)

        st.header("Кандидаты на новые характеристики")
        st.dataframe(new_df)

        st.header("Карта компетенций")
        knk_df_short = knk_df[knk_df.columns[3:-1]]
        main_comps = [item.name for item in full_char_list]
        for i in range(len(main_comps)):
            with st.expander(main_comps[i]):
                index_start = knk_df_short[
                    knk_df_short["Основная компетенция"] == main_comps[i]
                ].index[0]
                index_end = index_start
                while index_end < knk_df_short.shape[0] - 1:
                    index_end += 1
                    if len(knk_df_short.loc[index_end]["Основная компетенция"]) > 0:
                        break
                if index_end == knk_df_short.shape[0] - 1:
                    index_end += 1
                knk_df_short_selected = knk_df_short.loc[index_start : index_end - 1][
                    knk_df_short.columns[1:]
                ].reset_index(drop=True)
                st.markdown(
                    knk_df_short_selected.to_html(escape=False), unsafe_allow_html=True
                )
