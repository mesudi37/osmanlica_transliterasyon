# Ottoman Turkish Transliteration API

Modern Türkçe metni Osmanlı Arap harflerine (حروف عثمانیه) çeviren REST API.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # dosyayı düzenleyin
```

## Çalıştırma

```bash
uvicorn app.main:app --reload --port 8000
```

Sunucu hazır olduğunda **http://localhost:8000/docs** adresinden Swagger UI açılır.

---

## Endpoint'ler

### `GET /health`
Sunucu sağlık kontrolü.

```json
{ "status": "ok", "engine": "OttomanTransliterator", "version": "2.0.0" }
```

---

### `POST /transliterate`

Tek bir metni çevirir.

**İstek:**
```json
{
  "text": "vermemişlerdir.",
  "historical": true,
  "include_tokens": false
}
```

**Yanıt:**
```json
{
  "turkish":    "vermemişlerdir.",
  "ottoman":    "ویرممشلردر.",
  "confidence": 1.0,
  "tokens":     null
}
```

`include_tokens: true` ile token bazlı detay da döner:
```json
{
  "tokens": [
    {
      "token":   "vermemişlerdir",
      "ottoman": "ویرممشلردر",
      "source":  "tags",
      "debug":   "ver :: VERB+NEG+NARR+PLURAL+COPULA_ASSERT :: ver+memiş+ler+dir"
    }
  ]
}
```

`source` değerleri:

| Değer | Açıklama |
|-------|----------|
| `exact` | Sözlükten tam eşleşme |
| `override` | Elle yazılmış form |
| `tags` | Zeyrek morfolojik analizi |
| `english` | İngilizce transkripsiyon |
| `auto` | Basit harf haritası |
| `missing` | Çözümsüz (köşeli parantez ile işaretlenir) |
| `punct` | Noktalama |

---

### `POST /transliterate/batch`

Tek seferde en fazla 100 cümle.

**İstek:**
```json
{
  "items": [
    { "id": "s1", "text": "Merhaba dünya." },
    { "id": "s2", "text": "Nasılsınız?" }
  ],
  "historical": true,
  "include_tokens": false
}
```

**Yanıt:**
```json
{
  "results": [
    { "id": "s1", "turkish": "Merhaba dünya.", "ottoman": "...", "confidence": 0.9 },
    { "id": "s2", "turkish": "Nasılsınız?",    "ottoman": "...", "confidence": 1.0 }
  ]
}
```

---

## Ortam Değişkenleri

| Değişken | Varsayılan | Açıklama |
|----------|-----------|----------|
| `LOOKUP_FILE` | `manual_lookup.tsv` | Manuel sözlük dosyası |
| `ABBREV_FILE` | `abbrev_lookup.tsv` | Kısaltmalar sözlüğü |
| `HISTORICAL_ORTHOGRAPHY` | `true` | Osmanlıca yazım kuralları |
| `CORS_ORIGINS` | `*` | İzin verilen origin'ler (virgülle ayrılmış) |
| `OTTOMAN_BASE_DIR` | — | Veri dosyalarının dizini |

---

## Örnek cURL

```bash
# Tek cümle
curl -X POST http://localhost:8000/transliterate \
     -H "Content-Type: application/json" \
     -d '{"text": "Güneş doğudan doğar.", "include_tokens": true}'

# Toplu
curl -X POST http://localhost:8000/transliterate/batch \
     -H "Content-Type: application/json" \
     -d '{
       "items": [
         {"id":"1","text":"Merhaba."},
         {"id":"2","text":"Nasılsınız?"}
       ]
     }'
```
