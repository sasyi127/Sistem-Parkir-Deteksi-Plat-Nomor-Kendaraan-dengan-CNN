import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for, session, send_file
from keras.models import load_model
from keras.utils import load_img, img_to_array
from werkzeug.utils import secure_filename
from flaskext.mysql import MySQL
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
import numpy as np
import tensorflow as tf
import os
import cv2
import locale

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = '12345'

mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'db_histori'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
conn = mysql.connect()

model = load_model('CNN.h5')


def model_predict(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (int(img.shape[1]*.4), int(img.shape[0]*.4)))
    # convert bgr to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    img_opening = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
    img_norm = img_gray - img_opening
    (thresh, img_norm_bw) = cv2.threshold(
        img_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    (thresh, img_without_norm_bw) = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_vehicle, hierarchy = cv2.findContours(
        img_norm_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_plate_candidate = []
    index_counter_contour_vehicle = 0
    for contour_vehicle in contours_vehicle:
        x, y, w, h = cv2.boundingRect(contour_vehicle)
        aspect_ratio = w/h
        if w >= 200 and aspect_ratio <= 4:
            index_plate_candidate.append(index_counter_contour_vehicle)
        index_counter_contour_vehicle += 1
    img_show_plate = img.copy()
    img_show_plate_bw = cv2.cvtColor(img_norm_bw, cv2.COLOR_GRAY2RGB)

    if len(index_plate_candidate) == 0:
        return ("Plat nomor tidak ditemukan")
    elif len(index_plate_candidate) == 1:
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(
            contours_vehicle[index_plate_candidate[0]])
        cv2.rectangle(img_show_plate, (x_plate, y_plate),
                      (x_plate+w_plate, y_plate+h_plate), (0, 255, 0), 5)
        cv2.rectangle(img_show_plate_bw, (x_plate, y_plate),
                      (x_plate+w_plate, y_plate+h_plate), (0, 255, 0), 5)
        img_plate_gray = img_gray[y_plate:y_plate +
                                  h_plate, x_plate:x_plate+w_plate]
    else:
        x_plate, y_plate, w_plate, h_plate = cv2.boundingRect(
            contours_vehicle[index_plate_candidate[1]])
        cv2.rectangle(img_show_plate, (x_plate, y_plate),
                      (x_plate+w_plate, y_plate+h_plate), (0, 255, 0), 5)
        cv2.rectangle(img_show_plate_bw, (x_plate, y_plate),
                      (x_plate+w_plate, y_plate+h_plate), (0, 255, 0), 5)
        img_plate_gray = img_gray[y_plate:y_plate +
                                  h_plate, x_plate:x_plate+w_plate]
        print('Dapat dua lokasi plat, pilih lokasi plat kedua')
    (thresh, img_plate_bw) = cv2.threshold(
        img_plate_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    img_plate_bw = cv2.morphologyEx(
        img_plate_bw, cv2.MORPH_OPEN, kernel)
    contours_plate, hierarchy = cv2.findContours(
        img_plate_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    index_chars_candidate = []
    index_counter_contour_plate = 0
    img_plate_rgb = cv2.cvtColor(img_plate_gray, cv2.COLOR_GRAY2BGR)
    img_plate_bw_rgb = cv2.cvtColor(img_plate_bw, cv2.COLOR_GRAY2RGB)
    for contour_plate in contours_plate:
        x_char, y_char, w_char, h_char = cv2.boundingRect(contour_plate)
        if h_char >= 40 and h_char <= 60 and w_char >= 10:
            index_chars_candidate.append(index_counter_contour_plate)
            cv2.rectangle(img_plate_rgb, (x_char, y_char),
                          (x_char+w_char, y_char+h_char), (0, 255, 0), 5)
            cv2.rectangle(img_plate_bw_rgb, (x_char, y_char),
                          (x_char+w_char, y_char+h_char), (0, 255, 0), 5)

        index_counter_contour_plate += 1

    if index_chars_candidate == []:
        return ('Karakter tidak tersegmentasi')
    else:
        score_chars_candidate = np.zeros(len(index_chars_candidate))
        counter_index_chars_candidate = 0
        for chars_candidateA in index_chars_candidate:
            xA, yA, wA, hA = cv2.boundingRect(contours_plate[chars_candidateA])
            for chars_candidateB in index_chars_candidate:
                if chars_candidateA == chars_candidateB:
                    continue
                else:
                    xB, yB, wB, hB = cv2.boundingRect(
                        contours_plate[chars_candidateB])
                    y_difference = abs(yA - yB)
                    if y_difference < 11:
                        score_chars_candidate[counter_index_chars_candidate] = score_chars_candidate[counter_index_chars_candidate] + 1
            counter_index_chars_candidate += 1
        index_chars = []
        chars_counter = 0
        for score in score_chars_candidate:
            if score == max(score_chars_candidate):
                index_chars.append(index_chars_candidate[chars_counter])
            chars_counter += 1
        img_plate_rgb2 = cv2.cvtColor(img_plate_gray, cv2.COLOR_GRAY2BGR)
        for char in index_chars:
            x, y, w, h = cv2.boundingRect(contours_plate[char])
            cv2.rectangle(img_plate_rgb2, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(img_plate_rgb2, str(index_chars.index(char)),
                        (x, y + h + 50), cv2.FONT_ITALIC, 2.0, (0, 0, 255), 3)
    x_coors = []
    for char in index_chars:
        x, y, w, h = cv2.boundingRect(contours_plate[char])
        x_coors.append(x)
    x_coors = sorted(x_coors)

    # untuk menyimpan karakter
    index_chars_sorted = []
    for x_coor in x_coors:
        for char in index_chars:
            x, y, w, h = cv2.boundingRect(contours_plate[char])
            if x_coors[x_coors.index(x_coor)] == x:
                index_chars_sorted.append(char)
    img_plate_rgb3 = cv2.cvtColor(img_plate_gray, cv2.COLOR_GRAY2BGR)
    for char_sorted in index_chars_sorted:
        x, y, w, h = cv2.boundingRect(contours_plate[char_sorted])
        cv2.rectangle(img_plate_rgb3, (x, y), (x+w, y+h), (0, 255, 0), 5)
        cv2.putText(img_plate_rgb3, str(index_chars_sorted.index(
            char_sorted)), (x, y + h + 50), cv2.FONT_ITALIC, 2.0, (0, 0, 255), 3)
    img_height = 40
    img_width = 40

    # klas karakter
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
                   'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    num_plate = []

    for char_sorted in index_chars_sorted:
        x, y, w, h = cv2.boundingRect(contours_plate[char_sorted])

        # potong citra karakter
        char_crop = cv2.cvtColor(
            img_plate_bw[y:y+h, x:x+w], cv2.COLOR_GRAY2BGR)
        char_crop = cv2.resize(char_crop, (img_width, img_height))
        img_array = img_to_array(char_crop)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        num_plate.append(class_names[np.argmax(score)])

    plate_number = ''
    for a in num_plate:
        plate_number += a

    return plate_number


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower(
           ) in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])
    else:
        return redirect(url_for('login'))


@app.route('/admin', methods=['GET'])
def admin():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin")
    data_admin = cursor.fetchall()
    locale.setlocale(locale.LC_TIME, 'id_ID')
    current_datetime = datetime.now().strftime("%A %d %B %Y")

    return render_template('admin.html', admin=data_admin, tanggal=current_datetime)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Cek apakah username ada di database
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM admin WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user:
            # Verifikasi password
            if user[2] == password:
                session['username'] = username
                return redirect(url_for('index'))
            else:
                error = 'Password salah'
        else:
            error = 'Username tidak ditemukan'

        # Jika ada kesalahan, tampilkan pesan error pada halaman login
        return render_template('login.html', error=error)

    return render_template('login.html')


@app.route('/registrasi', methods=['GET', 'POST'])
def registrasi():
    if request.method == 'POST':
        try:
            username = request.form['username']
            password = request.form['password']
            nama = request.form['nama']
            tgl_lahir = request.form['tgl_lahir']
            jenis_kelamin = request.form['jenis_kelamin']
            alamat = request.form['alamat']
            no_tlp = request.form['no_tlp']

            cursor = conn.cursor()
            cursor.execute(
                f"INSERT INTO admin (username, password, nama, tgl_lahir, jenis_kelamin, alamat, no_tlp) VALUES ('{username}', '{password}', '{nama}', '{tgl_lahir}', '{jenis_kelamin}', '{alamat}', '{no_tlp}')")
            conn.commit()
            conn.close()

            # Ganti 'login' dengan rute halaman login
            return redirect(url_for('login'))

        except Exception as e:
            return "Terjadi kesalahan saat registrasi."

    return render_template('registrasi.html')


@app.route('/logout')
def logout():
    # Hapus session 'username' untuk logout
    session.pop('username', None)
    return redirect(url_for('login'))


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(url_for('result', status='gagal'))

    file = request.files['image']

    if file.filename == '':
        return redirect(url_for('result', status='gagal'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        plate_number = model_predict(file_path, model)
        terdeteksi = True if plate_number != "Plat nomor tidak ditemukan" and plate_number != "Karakter tidak tersegmentasi" else False

        # Set lokalisasi ke bahasa Indonesia
        locale.setlocale(locale.LC_TIME, 'id_ID')
        # Dapatkan tanggal dan waktu saat ini
        current_datetime = datetime.now()
        date_str = current_datetime.strftime("%A %d %B %Y")
        time_str = current_datetime.strftime("%H:%M:%S")

        session['result'] = plate_number
        session['filename'] = filename
        session['tanggal'] = date_str
        session['waktu'] = time_str

        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO deteksi (gambar, terdeteksi, hasil) VALUES ('{filename}', '(int{terdeteksi})', '{plate_number}')")
        cursor.execute(
            f"INSERT INTO masuk (Plat_Nomor, Tanggal_Masuk, Waktu_Masuk) VALUES ('{plate_number}', '{date_str}', '{time_str}')")
        conn.commit()

        return redirect('result', code=302)
    else:
        return redirect('result')


@app.route('/result', methods=['GET', 'POST'])
def result():
    if 'image_name' not in session or 'result' not in session:
        status = request.args.get('status')
        if status == 'gagal':
            return render_template('result.html', error=True)
        else:
            return render_template('result.html', error=False)

    # Set lokalisasi ke bahasa Indonesia
    locale.setlocale(locale.LC_TIME, 'id_ID')
    # Dapatkan tanggal dan waktu saat ini
    current_datetime = datetime.now()
    date_str = current_datetime.strftime("%Y-%m-%d")
    time_str = current_datetime.strftime("%H:%M:%S")
    session['tanggal'] = date_str
    session['waktu'] = time_str
    result = session['result']

    return render_template('result.html', result=result, datetime=current_datetime)


@app.route('/deteksi', methods=['GET'])
def deteksi():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM deteksi")
    deteksi_data = cursor.fetchall()
    locale.setlocale(locale.LC_TIME, 'id_ID')
    current_datetime = datetime.now().strftime("%A %d %B %Y")

    return render_template('deteksi.html', deteksi=deteksi_data, tanggal=current_datetime)


@app.route('/masuk', methods=['GET', 'POST'])
def masuk():
    if request.method == 'GET':
        kata_kunci = request.args.get('kata_kunci')

        if kata_kunci:
            # Query database untuk mendapatkan data laporan masuk sesuai dengan kata kunci
            cursor = conn.cursor()
            query = f"SELECT * FROM masuk WHERE Tanggal_Masuk LIKE '%{kata_kunci}%'"
            cursor.execute(query)
            data_masuk = cursor.fetchall()
            cursor.close()
        else:
            # Ambil semua data laporan masuk jika tidak ada filter kata kunci
            cursor = conn.cursor()
            query = "SELECT * FROM masuk"
            cursor.execute(query)
            data_masuk = cursor.fetchall()
            cursor.close()

    locale.setlocale(locale.LC_TIME, 'id_ID')
    current_datetime = datetime.now().strftime("%A %d %B %Y")

    return render_template('masuk.html', masuk=data_masuk, tanggal=current_datetime, kata_kunci=kata_kunci)


@app.route('/form_keluar', methods=['GET', 'POST'])
def form_keluar():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            id_masuk = int(request.form['id_masuk'])
            cursor = conn.cursor()

            # Pemeriksaan apakah ID_Masuk sudah ada di database keluar
            cursor.execute(
                f"SELECT * FROM keluar WHERE ID_Masuk = {id_masuk}"
            )
            data_keluar = cursor.fetchone()
            if data_keluar:
                return "ID_Masuk Sudah Keluar dari Parkir."

            # Set lokalisasi ke bahasa Indonesia
            locale.setlocale(locale.LC_TIME, 'id_ID')

            current_datetime = datetime.now()
            tanggal_keluar = current_datetime.strftime("%A %d %B %Y")
            waktu_keluar = current_datetime.strftime("%H:%M:%S")

            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM masuk WHERE ID_Masuk = {id_masuk}"
            )
            data_masuk = cursor.fetchone()
            if not data_masuk:
                return "ID_Masuk Tidak Ditemukan di Database."

            tanggal_masuk_str = data_masuk[2]
            waktu_masuk_str = data_masuk[3]

            # Ubah menjadi objek datetime
            tanggal_masuk = datetime.strptime(
                tanggal_masuk_str, "%A %d %B %Y")
            waktu_masuk = datetime.strptime(waktu_masuk_str, "%H:%M:%S")

            # Gabungkan tanggal dan waktu menjadi satu objek datetime
            tanggal_waktu_masuk = datetime.combine(
                tanggal_masuk.date(), waktu_masuk.time())

            # Hitung durasi_parkir in hours
            durasi_parkir_seconds = (current_datetime -
                                     tanggal_waktu_masuk).seconds
            durasi_parkir_hours = durasi_parkir_seconds // 3600

            # Set harga parkir tetap 2000 jika durasi kurang dari 1 jam
            if durasi_parkir_hours < 1:
                harga_parkir = 2000
            else:
                harga_parkir = 2000 * durasi_parkir_hours

            cursor.execute(
                f"INSERT INTO keluar (ID_Masuk, Tanggal_Keluar, Waktu_Masuk, Waktu_Keluar, Durasi_Parkir, Harga_Parkir) VALUES ('{id_masuk}', '{tanggal_keluar}', '{waktu_masuk_str}', '{waktu_keluar}', {durasi_parkir_hours}, {harga_parkir})")
            conn.commit()

            return redirect(url_for('keluar'))

        except Exception as e:
            print(f"Error: {e}")
            return "Terjadi kesalahan saat memproses permintaan."

    else:
        return render_template('form_keluar.html')


@app.route('/keluar', methods=['GET', 'POST'])
def keluar():
    if request.method == 'GET':
        kata_kunci = request.args.get('kata_kunci')

        if kata_kunci:
            # Query database untuk mendapatkan data laporan masuk sesuai dengan kata kunci
            cursor = conn.cursor()
            query = f"SELECT * FROM keluar WHERE Tanggal_Keluar LIKE '%{kata_kunci}%'"
            cursor.execute(query)
            data_keluar = cursor.fetchall()
            cursor.close()
        else:
            # Ambil semua data laporan masuk jika tidak ada filter kata kunci
            cursor = conn.cursor()
            query = "SELECT * FROM keluar"
            cursor.execute(query)
            data_keluar = cursor.fetchall()
            cursor.close()
    locale.setlocale(locale.LC_TIME, 'id_ID')
    current_datetime = datetime.now().strftime("%A %d %B %Y")

    return render_template('keluar.html', keluar=data_keluar, tanggal=current_datetime, kata_kunci=kata_kunci)


@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')


@app.route('/laporan', methods=['GET'])
def laporan():
    return render_template('laporan.html')


@app.route('/download_laporan_masuk', methods=['GET'])
def download_laporan_masuk():
    kata_kunci = request.args.get('kata_kunci')
    cursor = conn.cursor()
    if kata_kunci:
        query = f"SELECT * FROM masuk WHERE Tanggal_Masuk LIKE '%{kata_kunci}%'"
        cursor.execute(query)
        data_masuk = cursor.fetchall()
    else:
        cursor.execute("SELECT * FROM masuk")
        data_masuk = cursor.fetchall()

    # Buat buffer BytesIO untuk menyimpan konten PDF
    buffer = BytesIO()

    # Buat dokumen PDF menggunakan ReportLab
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    # Kontainer untuk elemen 'Flowable' dalam PDF
    elements = []

    # Tambahkan judul dan konten lainnya ke PDF
    styles = getSampleStyleSheet()
    title = Paragraph("Laporan Data Kendaraan Masuk", styles['Title'])
    elements.append(title)

    # Buat tabel untuk menampilkan data
    table_data = [['ID Kendaraan',
                   'Plat Nomor Kendaraan', 'Tanggal Masuk', 'Waktu Masuk']]
    for data in data_masuk:
        table_data.append([str(data[0]), str(data[1]), str(data[2]), str(
            data[3])])

    table = Table(table_data)
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='laporan_kendaraan_masuk.pdf', mimetype='application/pdf')


@app.route('/download_laporan_keluar', methods=['GET'])
def download_laporan_keluar():
    kata_kunci = request.args.get('kata_kunci')
    cursor = conn.cursor()
    if kata_kunci:
        query = f"SELECT * FROM keluar WHERE Tanggal_Keluar LIKE '%{kata_kunci}%'"
        cursor.execute(query)
        data_keluar = cursor.fetchall()
    else:
        cursor.execute("SELECT * FROM keluar")
        data_keluar = cursor.fetchall()

    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("Laporan Data Kendaraan Keluar", styles['Title'])
    elements.append(title)

    table_data = [['No.', 'ID Masuk', 'Tanggal Keluar', 'Waktu Masuk',
                   'Waktu Keluar', 'Durasi Parkir (menit)', 'Harga Parkir (Rp)']]
    for data in data_keluar:
        table_data.append([str(data[0]), str(data[1]), str(data[2]), str(
            data[3]), str(data[4]), str(data[5]), str(data[6])])

    table = Table(table_data)
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='laporan_kendaraan_keluar.pdf', mimetype='application/pdf')


@app.route('/download_data_admin', methods=['GET'])
def download_data_admin():
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admin")
    data_admin = cursor.fetchall()

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)

    elements = []

    styles = getSampleStyleSheet()
    title = Paragraph("Laporan Data Admin", styles['Title'])
    elements.append(title)

    table_data = [['No.', 'Nama', 'Tanggal Lahir', 'Jenis Kelamin',
                   'Alamat', 'No Handphone']]
    for data in data_admin:
        table_data.append([str(data[0]), str(data[3]), str(
            data[4]), str(data[5]), str(data[6]), str(data[7])])

    table = Table(table_data)
    elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name='laporan_data_admin.pdf', mimetype='application/pdf')


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.debug = False
    app.run(host='0.0.0.0', port=5000)
