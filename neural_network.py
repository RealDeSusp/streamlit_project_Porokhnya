import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.python.keras.optimizer_v1 import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
import streamlit as st
import os


matplotlib.use('TkAgg')
tf.compat.v1.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.set_option('deprecation.showPyplotGlobalUse', False)


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# удаляем столбец DEATH_EVENT, поскольку он сильно затрудняет прогнозирование
data.drop('DEATH_EVENT', axis=1, inplace=True)
data = data[data['creatinine_phosphokinase'] <= 1500]
data = data[data['serum_creatinine'] <= 4]

# Разделение данных на признаки (X) и целевую переменную (y)
X = data.drop('anaemia', axis=1).values
y = data['anaemia'].values

# Разделение данных на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Стандартизация данных
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети с другими параметрами оптимизатора и функцией потерь
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1], kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Компиляция модели с другим оптимизатором и функцией потерь
optimizer = Adam(lr=0.0001)  # Изменяем learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели на тестовых данных
loss, accuracy = model.evaluate(X_test, y_test)

# Создание модели нейронной сети с дополнительными слоями и/или большим количеством нейронов
model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=X_train.shape[1], kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(units=64, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(units=32, activation='relu', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели с большим количеством эпох
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Оценка модели на тестовых данных
model.evaluate(X_test, y_test)


def visualize_data():
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    # удаляем столбец DEATH_EVENT, поскольку он сильно затрудняет прогнозирование
    data.drop('DEATH_EVENT', axis=1, inplace=True)
    data = data[data['creatinine_phosphokinase'] <= 1500]
    data = data[data['serum_creatinine'] <= 4]

    # Построение графиков для переменных с непрерывными значениями
    plt.figure(figsize=(15, 15))

    # Распределение возраста
    plt.subplot(3, 2, 1)
    sns.histplot(data['age'], bins=20, kde=True)
    plt.title('Распределение возраста, г.')
    plt.xlabel('')
    plt.ylabel('')

    # Распределение уровня креатинкиназы
    plt.subplot(3, 2, 2)
    sns.histplot(data['creatinine_phosphokinase'], bins=20, kde=True)
    plt.title('Распределение уровня креатинкиназы, мкг/л.')
    plt.xlabel('')
    plt.ylabel('')

    # Распределение процента выброса
    plt.subplot(3, 2, 3)
    sns.histplot(data['ejection_fraction'], bins=20, kde=True)
    plt.title('Распределение процента выброса, %.')
    plt.xlabel('')
    plt.ylabel('')

    # Распределение тромбоцитов
    plt.subplot(3, 2, 4)
    sns.histplot(data['platelets'], bins=20, kde=True)
    plt.title('Распределение тромбоцитов, килотромбоциты/мл.')
    plt.xlabel('')
    plt.ylabel('')

    # Распределение сывороточного креатинина
    plt.subplot(3, 2, 5)
    sns.histplot(data['serum_creatinine'], bins=20, kde=True)
    plt.title('Распределение сывороточного креатинина, мг/дл.')
    plt.xlabel('')
    plt.ylabel('')

    # Распределение сывороточного натрия
    plt.subplot(3, 2, 6)
    sns.histplot(data['serum_sodium'], bins=20, kde=True)
    plt.title('Распределение сывороточного натрия, мэкв/л.')
    plt.xlabel('')
    plt.ylabel('')

    plt.tight_layout(pad=3.0)

    # Вывод графиков с помощью Streamlit
    st.pyplot()

    # Создаем новую фигуру для гистограммы "Количество дней наблюдения"
    plt.figure(figsize=(10, 5))

    # Распределение времени наблюдения
    sns.histplot(data['time'], bins=20, kde=True)
    plt.title('Количество дней наблюдения, д.')
    plt.xlabel('')
    plt.ylabel('')

    plt.tight_layout()

    # Вывод графика с помощью Streamlit
    st.pyplot()

    # Графики для переменных с двумя значениями
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 3, 1)
    data['anaemia'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Наличие анемии')
    plt.ylabel('')

    plt.subplot(2, 3, 2)
    data['diabetes'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Наличие диабета')
    plt.ylabel('')

    plt.subplot(2, 3, 3)
    data['high_blood_pressure'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Наличие повышенного давления')
    plt.ylabel('')

    plt.subplot(2, 3, 4)
    data['sex'].replace({0: 'Женщины', 1: 'Мужчины'}).value_counts().plot(kind='pie', autopct='%1.1f%%',
                                                                          colors=['skyblue', 'lightcoral'])
    plt.title('Пол')
    plt.ylabel('')

    plt.subplot(2, 3, 5)
    data['smoking'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    plt.title('Курение')
    plt.ylabel('')

    plt.tight_layout()

    # Вывод графиков с помощью Streamlit
    st.pyplot()


def correlation_analysis():
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    # удаляем столбец DEATH_EVENT, поскольку он сильно затрудняет прогнозирование
    data.drop('DEATH_EVENT', axis=1, inplace=True)
    data = data[data['creatinine_phosphokinase'] <= 1500]
    data = data[data['serum_creatinine'] <= 4]

    # Названия переменных
    column_names = {
        'age': 'Возраст, г.',
        'anaemia': 'Наличие анемии',
        'creatinine_phosphokinase': 'Уровень креатинкиназы, мкг/л.',
        'diabetes': 'Наличие диабета',
        'ejection_fraction': 'Процент выброса, %',
        'high_blood_pressure': 'Наличие повышенного давления',
        'platelets': 'Тромбоциты, килотромбоциты/мл.',
        'serum_creatinine': 'Сывороточный креатинин, мг/дл.',
        'serum_sodium': 'Сывороточный натрий, мэкв/л.',
        'sex': 'Пол',
        'smoking': 'Курение',
        'time': 'Количество дней наблюдения, д.'
    }

    # Отображаем тепловую карту корреляции
    plt.figure(figsize=(15, 12))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=2, annot_kws={"size": 12}, square=True)
    plt.title('Тепловая карта корреляции')
    plt.xticks(rotation=45, ha='right')  # Наклоняем названия переменных на оси x под углом 45 градусов
    plt.yticks(rotation=0)  # Убираем наклон названий переменных на оси y
    plt.tight_layout(rect=(0, 0, 0.95, 1))  # Добавляем отступы справа
    plt.subplots_adjust(right=0.95)

    # Изменяем подписи переменных на графике
    plt.gca().set_xticklabels([column_names[col] for col in data.columns], rotation=45, ha='right')
    plt.gca().set_yticklabels([column_names[col] for col in data.columns], rotation=0)

    # Выводим график с помощью Streamlit
    st.pyplot()


def prediction_page():
    st.title("Прогнозирование наличия анемии у пациента")

    # Загрузка данных
    data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    # удаляем столбец DEATH_EVENT, поскольку он сильно затрудняет прогнозирование
    data.drop('DEATH_EVENT', axis=1, inplace=True)
    data = data[data['creatinine_phosphokinase'] <= 1500]
    data = data[data['serum_creatinine'] <= 4]

    # Получение списка первых 30 пациентов
    patients = [f"Пациент {i}" for i in range(1, 31)]

    # Выпадающий список с выбором пациента
    selected_patient = st.selectbox("Выберите пациента", patients)

    # Индекс выбранного пациента
    patient_index = int(selected_patient.split()[1]) - 1

    # Кнопка для прогнозирования
    if st.button("Прогнозировать наличие анемии у пациента"):
        # Вызываем функцию predict_anaemia для выбранного пациента
        probability_of_anaemia_percent, prediction_level = predict_anaemia(data, patient_index)

        # Вывод результатов прогноза
        st.write(f"Вероятность наличия анемии у пациента: {probability_of_anaemia_percent}%")
        st.write(f"Прогноз: {prediction_level}")


def predict_anaemia(data, patient_index):
    # Выбираем данные для выбранного пациента
    patient_data = data.drop('anaemia', axis=1).iloc[patient_index].values.reshape(1, -1)

    # Стандартизация данных пациента
    patient_data = scaler.transform(patient_data)

    # Предсказание вероятности наличия анемии у пациента
    probability_of_anaemia = model.predict(patient_data)[0][0]
    probability_of_anaemia_percent = round(probability_of_anaemia * 100, 2)

    # Классификация пациента по уровням
    if probability_of_anaemia_percent < 20:
        prediction_level = "Пациенту, вероятнее всего, не нужно лечение от анемии."
    elif 20 <= probability_of_anaemia_percent < 40:
        prediction_level = "Пациенту стоит обратиться к врачу и проверить своё состояние здоровья."
    elif 40 <= probability_of_anaemia_percent < 80:
        prediction_level = "Пациенту, вероятнее всего, нужно лечение от анемии."
    else:
        prediction_level = "Пациенту нужно срочное лечение от анемии."

    return probability_of_anaemia_percent, prediction_level


def conclusions_page():
    st.title("Выводы")

    # Заглушка для вывода анализа исходных данных
    st.header("Анализ исходных данных")
    st.write("""
    **Распределение возраста:** Средний возраст пациентов составляет 60 лет, что подтверждает высокую среднюю
     возрастную группу с сердечной недостаточностью.

    **Наличие анемии:** Обнаружено у 44.4% пациентов, что свидетельствует о значительной распространенности этого
     состояния среди исследуемой группы.

    **Распределение уровня креатинкиназы:** Несмотря на среднее значение в районе 20-200, наблюдается наличие 
    значительного числа значений, превышающих 600, что может указывать на возможные осложнения.

    **Наличие диабета:** Обнаружено у 42.5% пациентов, что свидетельствует о высокой распространенности этого 
    фактора риска.

    **Распределение процента выброса:** Среднее значение приблизительно равно 38, что соответствует нормальному 
    значению, указывая на относительно стабильную функцию сердца у большинства пациентов.

    **Наличие повышенного давления:** Обнаружено у 36.1% пациентов, что подчеркивает важность мониторинга 
    артериального давления у этой группы.

    **Распределение тромбоцитов:** Среднее значение объема тромбоцитов в крови пациентов составляет 290000, 
    что находится в нормальном диапазоне. Однако следует отметить, что существует значительное количество 
    пациентов, чьи значения не попадают в это среднее, поэтому этот показатель также важно учитывать при анализе.

    **Распределение сывороточного креатина:** Среднее значение равно 1.1, что может указывать на некоторую 
    степень повышения уровня сывороточного креатина у части пациентов.

    **Распределение сывороточного натрия:** Средний уровень сывороточного натрия составляет 137, что соответствует 
    нормальным показателям. Но также важно учитывать тех пациентов, чьи значения не соответствуют этому среднему, 
    чтобы получить более полное представление о состоянии их здоровья.

    **Пол:** 63.9% пациентов - мужчины, что может указывать на более высокую предрасположенность мужчин к сердечной 
    недостаточности.

    **Курение:** 32.7% пациентов курят, что свидетельствует о значительной распространенности этого негативного 
    фактора среди исследуемой группы.

    **Распределение времени:** Наблюдаются два сосредоточения вокруг средних значений в 90 и 210 днях, что может 
    указывать на различные характеристики процесса лечения и прогноза у пациентов.
    """)

    # Выводы из корреляционного анализа
    st.header("Выводы из корреляционного анализа")
    st.write("""
    **Возраст:** Возраст положительно коррелирует с уровнем сывороточного креатинина (0.08) и с сывороточным натрием 
    (0.10), что может указывать на увеличение уровня этих показателей с возрастом. .

    **Распределение уровня креатинкиназы:** Наблюдается слабая положительная корреляция с объемом тромбоцитов 
    (0.20), что может указывать на связь повышенного уровня креатинкиназы с увеличением объема тромбоцитов в крови. 
    Также выявляется слабая отрицательная корреляция со временем (-0.22), указывающая на возможность значительных
     изменений уровня креатинкиназы с течением времени.

    **Наличие диабета:** Обнаруживается слабая отрицательная корреляция с тромбоцитами (-0.19) и временем 
    (-0.13), что может указывать на изменение уровня тромбоцитов и креатинкиназы у пациентов с диабетом с течением 
    времени.

    **Распределение процента выброса:** Наблюдается положительная корреляция с возрастом (0.06) и полом пациента 
    (0.08), что может указывать на более высокий процент выброса крови у пожилых пациентов и мужчин.

    **Наличие повышенного давления:** Выявляется слабая отрицательная корреляция с уровнем сывороточного натрия 
    (-0.09) и временем (-0.04), что может указывать на изменения уровня сывороточного натрия и креатинкиназы у пациентов 
    с повышенным давлением с течением времени.

    **Объем тромбоцитов:** Обнаруживается слабая положительная корреляция с возрастом (0.08) и полом (0.01), что 
    может указывать на более высокий уровень тромбоцитов у пожилых пациентов-мужчин.

    **Уровень сывороточного креатинина:** Выявляется слабая отрицательная корреляция со временем (-0.05), что может 
    указывать на снижение уровня сывороточного креатинина у пациентов с течением времени.

    **Уровень сывороточного натрия:** Наблюдается слабая положительная корреляция с возрастом (0.20) и с диабетом 
    (0.07), что может указывать на более высокий уровень сывороточного натрия у пожилых пациентов и пациентов с 
    диабетом.

    **Пол:** Обнаруживается слабая положительная корреляция с возрастом (0.06) и с диабетом (0.07), что может указывать 
    на более высокий уровень диабета среди более взрослых мужчин.

    **Курение:** Наблюдается слабая положительная корреляция с возрастом (0.10), что может указывать на более глубокий 
    возраст среди курильщиков.

    **Распределение времени:** Выявляется слабая отрицательная корреляция с диабетом (-0.22) и с повышенным давлением 
    (-0.13), что может указывать на меньшее время наблюдения у пациентов с диабетом и повышенным давлением.
    """)


def main():
    st.title("Система прогнозирования наличия анемии у пациентов")
    # Создаем боковую панель для навигации
    with st.sidebar:
        st.title("Навигация")
        choice = st.radio("", ["Главная", "Визуализация исходных данных", "Корреляционный анализ",
                               "Прогнозирование наличия анемии", "Выводы"])
    # Основная часть страницы
    if choice == "Главная":
        st.write("Добро пожаловать в систему прогнозирования наличия анемии у пациентов!")
    elif choice == "Визуализация исходных данных":
        st.title("Визуализация исходных данных")
        visualize_data()
    elif choice == "Корреляционный анализ":
        st.title("Корреляционный анализ")
        correlation_analysis()
    elif choice == "Прогнозирование наличия анемии":
        prediction_page()
    elif choice == "Выводы":
        conclusions_page()


if __name__ == "__main__":
    main()
