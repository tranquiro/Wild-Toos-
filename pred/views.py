from django.shortcuts import render
# Create your views here.
from django.views.generic import TemplateView
from .forms import ImageForm
from .main import detect

class PredView(TemplateView):
    # コンストラクタ
    def __init__(self):
        self.params = {'result_list': [],
                       'result_name': "",
                       'result_img': "",
                       'form': ImageForm()}

    # GETリクエスト（index.htmlを初期表示）
    def get(self, req):
        return render(req, 'pred/index.html', self.params)

    # POSTリクエスト（index.htmlに結果を表示）
    def post(self, req):
        # POSTされたフォームデータを取得
        form = ImageForm(req.POST, req.FILES)
        # フォームデータのエラーチェック
        if not form.is_valid():
            raise ValueError('invalid form')
        # フォームデータから画像ファイルを取得
        image = form.cleaned_data['image']
        # 画像ファイルを指定して顔分類
        result = detect(image)
        # 顔分類の結果を格納
        self.params['result_list'], self.params['result_name'], self.params['result_img'] = result
        # ページの描画指示
        return render(req, 'pred/index.html', self.params)