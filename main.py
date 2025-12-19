import re

def main():
    with open("./data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))

    sample = raw_text[100:199]

    print(sample)

    result = re.split(r'([,.]|\s)', sample)
    result = [item for item in result if item.strip()]
    print(result)


if __name__ == "__main__":
    main()
