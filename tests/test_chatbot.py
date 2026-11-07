from src.chatbot import get_bot_response

TESTS = [
    ("Hola, buenas tardes", "saludo"),
    ("Quiero comprar una moto", "buscar_moto"),
    ("Busco una moto que gaste poca gasolina", "moto_economica"),
    ("Necesito una moto para trabajar todos los dias", "moto_trabajo"),
    ("La usare solo en ciudad", "moto_ciudad"),
    ("Quiero viajar por caminos de tierra", "moto_aventura"),
    ("Cuanto cuesta la moto", "consultar_precio"),
    ("Puedo pagar a cuotas", "consultar_credito"),
    ("Que modelos tienen disponibles", "consultar_modelos"),
    ("Quiero comparar dos modelos", "comparar_modelos"),
    ("Necesito hablar con un asesor", "contacto_asesor"),
    ("Gracias por la ayuda", "despedida"),
    ("Busco algo economico para ir al trabajo", "moto_economica"),
    ("Quiero una moto para delivery", "moto_trabajo"),
    ("Necesito una moto para carretera y montaña", "moto_aventura"),
    ("Tienen financiamiento", "consultar_credito"),
    ("Dime los modelos de HERO", "consultar_modelos"),
    ("Cual me conviene mas entre dos motos", "comparar_modelos"),
    ("Pasame con un vendedor", "contacto_asesor"),
    ("Hasta luego", "despedida")
]

if __name__ == "__main__":
    ok = 0
    print("\nRESULTADOS DE LAS 20 PRUEBAS\n")
    for i, (text, expected) in enumerate(TESTS, start=1):
        result = get_bot_response(text)
        pred = result["intent"]
        passed = pred == expected
        ok += int(passed)
        mark = "OK" if passed else "FALLO"
        print(
            f"{i:02d}. {mark} | Esperado: {expected:<18} "
            f"| Predicho: {pred:<18} | Confianza: {result['confidence']:.2%}"
        )

    print(f"\nPruebas correctas: {ok}/{len(TESTS)}")
    print(f"Tasa de éxito: {ok/len(TESTS):.2%}")
