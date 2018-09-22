from IPython.display import HTML

def display_images(image_paths):
    html_content = "<div width='100%'>"
    for image_path in image_paths:
        html_content += '<div style="font-size: 10px; display:inline-block; width: 224px; border:1px solid black">\
         {image_path}:\
         <img src="{image_path}" style="display:inline-block;"> </div>'.format(image_path=image_path)
    html_content += '</div>'
    display(HTML(html_content))