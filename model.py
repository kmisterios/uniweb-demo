import glob
import json
import os
from pathlib import Path
from typing import IO, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from data import Competence, Knowledge, MainCompetence
from utils import clean_start_number, load_doc

load_dotenv()


class KNK:
    def __init__(
        self,
        queries_list: List[str],
        model_name: str = "gpt-4o",
        emb_model_name: str = "text-embedding-3-large",
        materials_folder: str = "./materials",
        embeddings_folder: str = "./embeddings",
    ):
        logger.info("Initializing model")
        self.client = OpenAI(api_key=os.getenv("OPENAI_TOKEN"))
        self.queries_list = queries_list
        self.model_name = model_name
        self.emb_model_name = emb_model_name
        self.materials = self.load_materials(folder=materials_folder)
        self.ref_characteristics = self.get_ref_characteristics()
        self.ref_characteristics_embs = self.load_embeddings(folder=embeddings_folder)
        self.query_class_map = {
            "компетенции": Competence,
            "знания": Knowledge,
            "оснкомпетенции": MainCompetence,
        }

    def load_materials(self, folder: str):
        paths_materials = glob.glob(f"{folder}/*.json")
        materials = {}
        for path in paths_materials:
            with open(path, "r") as f:
                materials[Path(path).stem] = json.load(f)
        logger.info("Materials loaded")
        return materials

    def load_embeddings(self, folder: str):
        res = {}
        if not Path(folder).exists():
            os.mkdir(folder)
        for query in self.queries_list:
            path_emb = Path(folder) / f"{query}.npy"
            if not path_emb.exists():
                logger.info(f"Computing embeddings for {query}")
                embs = self.get_emb_list(self.ref_characteristics[query])
                np.save(str(path_emb), embs)
            embs = np.load(str(path_emb))
            res[query] = embs
        logger.info("Embeddings loaded")
        return res

    def find_info(
        self,
        tech_task: str,
        query: str,
        examples: Optional[List] = None,
        query_desc: Optional[str] = None,
    ):
        question = f"Какие {query} должны быть у специалиста из должностной инструкции?"
        prompt = f"Должностная инструкция:\n{tech_task}\nВопрос: {question}\nОтвет:"
        system_prompt = (
            f"Вы эксперт в найме персонала. Необходимо извлечь необходимые {query} из должностной инструкции, "
            f"как отдельное поле в json с ключом '{query}'. "
            f"Не нужно придумывать того, чего нет в должностной инструкции."
        )
        if query_desc is not None:
            system_prompt += f"\n{query_desc}"
        if examples is not None:
            examples_str = "\n".join(examples)
            system_prompt = system_prompt + f" Примеры {query}:\n{examples_str}"

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content

    def get_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.emb_model_name)
            .data[0]
            .embedding
        )

    def get_emb_list(self, texts_list):
        res = []
        for text in tqdm(texts_list):
            emb = np.array(self.get_embedding(text=text))
            res.append(emb)
        return np.array(res)

    def find_match_cosine(
        self, query, emb_ref, emb_query, all_competencies, comp_query
    ):
        similarities = cosine_similarity(emb_query, emb_ref)
        idices_most_similar = similarities.argmax(axis=1)
        similarities = similarities.max(axis=1).round(3)
        most_similar = np.array(all_competencies)[idices_most_similar]
        df_sim = pd.DataFrame(
            np.vstack([comp_query, most_similar, similarities]).T,
            columns=[f"Найденные {query}", f"Существующие {query}", "cosine_sim"],
        )
        return df_sim

    def get_ref_characteristics(self):
        res = {}
        for query in self.queries_list:
            material_keys = list(
                filter(lambda x: x.split("_")[1] == query, list(self.materials.keys()))
            )
            if len(material_keys) == 0:
                res[query] = []
                continue
            material_key = material_keys[0]
            ref_values = []
            for key in self.materials[material_key]:
                self.materials[material_key][key] = [
                    clean_start_number(item)
                    for item in self.materials[material_key][key]
                ]
                ref_values += self.materials[material_key][key]
            res[query] = ref_values
        return res

    def compose_match_dfs(self, text_task: str):
        query_descs = self.materials["query_descriptions"]
        dfs_match = {}
        for query in tqdm(self.queries_list):
            characteristics_list_str = self.find_info(
                text_task,
                query,
                examples=self.ref_characteristics[query],
                query_desc=query_descs[query],
            )
            characteristics_list = json.loads(characteristics_list_str)[query]
            if query != "Soft Skills":
                characteristics_embs = self.get_emb_list(characteristics_list)
                df_sim_query = self.find_match_cosine(
                    query,
                    self.ref_characteristics_embs[query],
                    characteristics_embs,
                    self.ref_characteristics[query],
                    characteristics_list,
                )
                dfs_match[query] = df_sim_query
            else:
                dfs_match[query] = characteristics_list
        return dfs_match

    def get_found_dict(
        self, dict_match: Dict, values_found: List[Union[str, Knowledge, Competence]]
    ):
        res_dict = {}
        for val in values_found:
            found = False
            val_str = val
            if type(val) != str:
                val_str = val.name
            val_str = val_str.strip()
            for key in dict_match:
                if val_str in dict_match[key]:
                    if key not in res_dict:
                        res_dict[key] = []
                    found = True
                    if val not in res_dict[key]:
                        res_dict[key].append(val)
                    break
            if found == False:
                print(val_str)
        return res_dict

    def clean_column(self, values: np.array):
        prev_value = values[0]
        for i in range(1, len(values)):
            if values[i] == prev_value:
                values[i] = ""
            else:
                prev_value = values[i]
        return values

    def create_item_df(self, main_comp: MainCompetence):
        rows = []
        for comp in main_comp.child:
            if len(comp.child) == 0:
                row = [main_comp.name, comp.name, "", ""]
                rows.append(row)
            for know in comp.child:
                if len(know.child) == 0:
                    row = [main_comp.name, comp.name, know.name, ""]
                    rows.append(row)
                for skill in know.child:
                    row = [main_comp.name, comp.name, know.name, skill]
                    rows.append(row)
        df = pd.DataFrame(
            np.vstack(rows),
            columns=["Основная компетенция", "Компетенция", "Знания", "Навыки"],
        )
        return df

    def create_knk_df(
        self,
        main_comp_list: List[MainCompetence],
        df_position: pd.DataFrame,
        soft_skills: List[str],
    ):
        all_dfs = []
        for maincomp in main_comp_list:
            df = self.create_item_df(maincomp)
            all_dfs.append(df)
        df_all = pd.concat(all_dfs).reset_index(drop=True)
        for col in df_all.columns[:-1]:
            df_all[col] = self.clean_column(df_all[col].values)
        ## continue here
        position = df_position.iloc[0]["Существующие должности"]
        position_level = list(
            self.get_found_dict(
                self.materials["уровеньдолжности_должности"], [position]
            ).keys()
        )[0]
        df_all["Софт-скиллы"] = soft_skills + [""] * (
            df_all.shape[0] - len(soft_skills)
        )
        df_all["Направление (сфера)"] = ["Маркетинг"] + [""] * (df_all.shape[0] - 1)
        df_all["Должность"] = [position] + [""] * (df_all.shape[0] - 1)
        df_all["Уровень должности"] = [position_level] + [""] * (df_all.shape[0] - 1)
        df_all = df_all[list(df_all.columns[-3:]) + list(df_all.columns[:-3])]
        return df_all

    def create_new_charact_df(self, new_characts: Dict):
        max_len_new = max(
            list(map(len, [new_characts[query] for query in new_characts]))
        )
        cols = []
        queries = list(new_characts.keys())
        for query in queries:
            cols.append(
                list(new_characts[query])
                + [""] * (max_len_new - len(new_characts[query]))
            )
        return pd.DataFrame(np.vstack(cols).T, columns=queries)

    def create(self, file: IO[bytes]):
        text_task = load_doc(file)
        logger.info("DOC loaded")
        match_df_dict = self.compose_match_dfs(text_task=text_task)
        logger.info("Created match dfs")
        new_charact_dict = {}
        found_charact_dict = {}
        for query in self.queries_list:
            if query in ["Soft Skills", "должности"]:
                continue
            match_df_dict[query]["cosine_sim"] = match_df_dict[query][
                "cosine_sim"
            ].astype(float)
            perc_check = np.percentile(match_df_dict[query]["cosine_sim"].values, 7)
            new_charact = match_df_dict[query][
                match_df_dict[query]["cosine_sim"] < perc_check
            ][f"Найденные {query}"].values
            found_charact = match_df_dict[query][
                match_df_dict[query]["cosine_sim"] > perc_check
            ][f"Существующие {query}"].values
            found_charact_dict[query] = found_charact
            new_charact_dict[query] = new_charact

        full_char_list = None
        for query in self.queries_list:
            if query in ["Soft Skills", "должности"]:
                continue
            material_key = list(
                filter(lambda x: x.split("_")[1] == query, list(self.materials.keys()))
            )[0]
            query_parent = material_key.split("_")[0]
            found_parent = []
            if query_parent in found_charact_dict:
                found_parent = found_charact_dict[query_parent]
            characterstics = (
                found_charact_dict[query] if full_char_list is None else full_char_list
            )
            match_dict = self.get_found_dict(
                self.materials[material_key], characterstics
            )
            found_not_matched_str = list(
                set(found_parent) - set(found_parent).intersection(match_dict.keys())
            )
            found_not_matched = [
                self.query_class_map[query_parent](name=item, child=[])
                for item in found_not_matched_str
            ]
            found_from_child = [
                self.query_class_map[query_parent](name=item, child=match_dict[item])
                for item in match_dict
            ]
            full_char_list = found_not_matched + found_from_child

        knk_df = self.create_knk_df(
            main_comp_list=full_char_list,
            df_position=match_df_dict["должности"],
            soft_skills=match_df_dict["Soft Skills"],
        )
        new_df = self.create_new_charact_df(new_characts=new_charact_dict)
        logger.info("KNK df created")
        return knk_df, new_df, full_char_list


if __name__ == "__main__":
    queries = ["навыки", "знания", "компетенции", "должности", "Soft Skills"]
    f = open("./notebooks/data/долж_инструкция_маркетолог.docx", "rb")
    knk_creator = KNK(queries_list=queries)
    knk_df, new_df, full_char_list = knk_creator.create(f)
    print(knk_df.head(5))
    print(new_df.head(5))
    knk_df.to_csv("./knk_test.csv", index=False)
    new_df.to_csv("./new_df_test.csv", index=False)
