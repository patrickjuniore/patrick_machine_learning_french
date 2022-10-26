#https://jsonformatter.curiousconcept.com/#

#ValueError: If using all scalar values, you must pass an index

        logging.info("open report json")
        with open(report_json, "r", encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        #dict_data_report = json_data["data"][0]["region"]["cities"]
        dict_data_report = json_data["response"][0]["cases"]
        logging.info("wrote successfully into report json")
