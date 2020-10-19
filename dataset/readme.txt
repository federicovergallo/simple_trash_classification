The dataset is created from Google Images

- To do it first search what you need in the page, scroll down until reach the end then open the console and type:

urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));

This will generate a csv file with the URL of each image you have scrolled. Save the csv file in the folder "csv_files".

Then fill the params at the beginning of file "download.py" with the csv file you just downloaded and the destination folder.