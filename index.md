## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/wollbo/threshold/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```


<div class="layout-wrapper">
    <div class="controls">
        <label for="p">p:</label><input id="p" type="number" size="5" value="0.5" onchange="draw()" />
        <label for="lambda">lambda:</label><input id="lambda" type="number" size="5" value="1" onchange="draw()" />
    </div>
    <div id="renderer">
        <!-- here all the plots will be rendered -->
    </div>

    <link rel="stylesheet" href="/css/cost.css" />
    <script src="https://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script src="js/cost.js" charset="utf-8"></script>
</div>


For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/wollbo/threshold/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
