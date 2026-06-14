def get_quote():
    response = requests.get(
        "https://api.forismatic.com/api/1.0/",
        params={
            "method": "getQuote",
            "format": "json",
            "lang": "ru"
        }
    )
    return response.json()