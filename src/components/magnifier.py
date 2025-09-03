import streamlit.components.v1 as components

def magnifier(image_url: str, zoom: int = 2, size: int = 200):
    """
    마우스 돋보기(렌즈 확대)
    :param image_url: 이미지 경로
    :param zoom: 확대 배율
    :param size: 렌즈 지름(px)
    """
    html_code = f"""
    <style>
    .img-magnifier-container {{
      position: relative;
    }}
    .img-magnifier-glass {{
      position: absolute;
      border: 3px solid #000;
      border-radius: 50%;
      cursor: none;
      width: {size}px;
      height: {size}px;
    }}
    </style>
    <div class="img-magnifier-container">
      <img id="myimage" src="{image_url}" width="100%" />
      <div class="img-magnifier-glass" id="glass"></div>
    </div>
    <script>
    function magnify(imgID, zoom) {{
      var img, glass, w, h, bw;
      img = document.getElementById(imgID);
      glass = document.getElementById("glass");
      bw = 3; w = glass.offsetWidth / 2; h = glass.offsetHeight / 2;
      glass.style.backgroundImage = "url('" + img.src + "')";
      glass.style.backgroundRepeat = "no-repeat";
      glass.style.backgroundSize = (img.width * zoom) + "px " + (img.height * zoom) + "px";
      img.parentElement.addEventListener("mousemove", moveMagnifier);
      glass.addEventListener("mousemove", moveMagnifier);
      function moveMagnifier(e) {{
        var pos, x, y; e.preventDefault();
        pos = getCursorPos(e); x = pos.x; y = pos.y;
        glass.style.left = (x - w) + "px";
        glass.style.top = (y - h) + "px";
        glass.style.backgroundPosition = "-" + ((x * zoom) - w + bw) + "px -" + ((y * zoom) - h + bw) + "px";
      }}
      function getCursorPos(e) {{
        var a, x = 0, y = 0; e = e || window.event;
        a = img.getBoundingClientRect();
        x = e.pageX - a.left; y = e.pageY - a.top;
        x = x - window.pageXOffset; y = y - window.pageYOffset;
        return {{x:x, y:y}};
      }}
    }}
    magnify("myimage", {zoom});
    </script>
    """
    components.html(html_code, height=600, scrolling=False)
