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

        st.header("КНК")
        main_comps = [item.name for item in full_char_list]
        selected_mc_str = st.selectbox(
            "Основные компетенции",
            main_comps,
            index=None,
            placeholder="Выберите основную компетенцию",
        )
        if selected_mc_str:
            selected_item = list(
                filter(lambda x: x.name == selected_mc_str, full_char_list)
            )[0]
            comps = [item.name for item in selected_item.child]
            selected_c_str = st.selectbox(
                "Компетенции", comps, index=None, placeholder="Выберите компетенцию"
            )
            if selected_c_str:
                selected_item = list(
                    filter(lambda x: x.name == selected_c_str, selected_item.child)
                )[0]
                knowledge = [item.name for item in selected_item.child]
                if len(knowledge) != 0:
                    selected_know_str = st.selectbox(
                        "Знания", knowledge, index=None, placeholder="Выберите знание"
                    )
                    if selected_know_str:
                        selected_item = list(
                            filter(
                                lambda x: x.name == selected_know_str,
                                selected_item.child,
                            )
                        )[0]
                        skills = selected_item.child
                        if len(skills) != 0:
                            st.text("Навыки")
                            s = ""
                            for skill in skills:
                                s += "- " + skill + "\n"
                            st.markdown(s)
                        else:
                            st.markdown("**Для данного знания не были найдены умения**")
                else:
                    st.markdown("**Для данной компетенции не было найдено знаний**")
